from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from src.datasets import HeartDataset1D
from src.metrics import Accuracy, Recall
from loguru import logger

from torch import Tensor
import math

logger.remove()
logger.add("logs/logfile.log", level="DEBUG")

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        identity = x.clone()
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + identity)
        identity = x.clone()
        x = self.ff(x)
        x = self.layer_norm2(x + identity)
        return x

# Transformer Model for 1D Data
class TransformerModel1D(nn.Module):
    def __init__(self, config):
        super(TransformerModel1D, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden_size"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config["hidden_size"], dropout=config["dropout"], max_seq_len=config["max_seq_length"]
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config["hidden_size"], config["num_heads"], config["dropout"])
            for _ in range(config["num_layers"])
        ])
        self.fc = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1d(x.transpose(1, 2))
        x = self.positional_encoding(x.transpose(1, 2))
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def train(config: Dict):
    data_dir = Path(__file__).parent.parent.resolve() / "data"
    train_path = data_dir / "heart_big_train.parq"
    test_path = data_dir / "heart_big_test.parq"

    train_dataset = HeartDataset1D(train_path, target="target", apply_smote=True)
    test_dataset = HeartDataset1D(test_path, target="target")

    train_subset_size = int(0.5 * len(train_dataset))
    test_subset_size = int(0.5 * len(test_dataset))

    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    recall = Recall(average="macro")
    accuracy = Accuracy()
    model = TransformerModel1D(config)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0.0
    total_recall = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += loss_fn(outputs, y_batch).item()
            total_recall += recall(y_batch.cpu().numpy(), outputs.cpu().numpy())
            total_accuracy += accuracy(y_batch.cpu().numpy(), outputs.cpu().numpy())

    avg_recall = total_recall / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)

    session.report({
        "loss": test_loss / len(test_loader),
        "recall": avg_recall,
        "accuracy": avg_accuracy
    })

if __name__ == "__main__":
    ray.init(num_cpus=4)

    storage_path = f"file://{Path('hypertuning_results').resolve()}"

    config = {
        "input_size": 16,
        "max_seq_length": 12,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.2),
        "output_size": 5,
        "num_heads": tune.choice([2, 4, 8]),
        "num_layers": tune.randint(1, 6),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Recall")
    reporter.add_metric_column("Accuracy")

    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=5,
    )

    analysis = tune.run(
        train,
        config=config,
        metric="recall",
        mode="max",
        progress_reporter=reporter,
        num_samples=10,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
        storage_path=storage_path,
        resources_per_trial={
            "cpu": 4,
        }
    )

    print("Best configuration:", analysis.best_config)

    Path("hypertuning_results").mkdir(parents=True, exist_ok=True)

    results_df = analysis.results_df
    results_df.to_csv("hypertuning_results/results.csv", index=False)

    ray.shutdown()
