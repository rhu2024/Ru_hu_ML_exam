# from pathlib import Path
# from typing import Dict

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import numpy as np
# import ray
# from ray import tune
# from ray.air import session
# from ray.tune import CLIReporter
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.search.hyperopt import HyperOptSearch
# from src.datasets import HeartDataset2D
# from src.metrics import Accuracy, Recall
# from loguru import logger

# logger.remove()
# logger.add("logs/logfile.log", level="DEBUG")

# # ResidualBlock for 2D Convolutions
# class ResidualBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
#         super(ResidualBlock2D, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout2d(dropout)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# # CNN Model using ResidualBlock2D
# class CNNModel2D(nn.Module):
#     def __init__(self, config):
#         super(CNNModel2D, self).__init__()
#         self.block1 = ResidualBlock2D(1, config["hidden_size"], dropout=config["dropout"])
#         self.block2 = ResidualBlock2D(config["hidden_size"], config["hidden_size"] * 2, downsample=nn.Sequential(
#             nn.Conv2d(config["hidden_size"], config["hidden_size"] * 2, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(config["hidden_size"] * 2),
#         ), dropout=config["dropout"])
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(config["hidden_size"] * 2, config["output_size"])

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.pool(x).squeeze(-1).squeeze(-1)
#         x = self.fc(x)
#         return x

# def train(config: Dict):
#     # Gebruik het correcte pad naar de data-map
#     data_dir = Path(__file__).parent.parent.resolve() / "data"  # Map buiten /dev/
#     train_path = data_dir / "heart_big_train.parq"
#     test_path = data_dir / "heart_big_test.parq"

#     # Laad de data met HeartDataset2D
#     train_dataset = HeartDataset2D(train_path, target="target", shape=(16, 12), apply_smote=True)
#     test_dataset = HeartDataset2D(test_path, target="target", shape=(16, 12))

#     # Gebruik een percentage van de data
#     train_subset_size = int(0.5 * len(train_dataset))  # 50% van de trainingsdata
#     test_subset_size = int(0.5 * len(test_dataset))    # 50% van de testdata

#     train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
#     test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)

#     train_subset = Subset(train_dataset, train_indices)
#     test_subset = Subset(test_dataset, test_indices)

#     train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

#     recall = Recall(average="macro")
#     accuracy = Accuracy()
#     model = CNNModel2D(config)

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     loss_fn = nn.CrossEntropyLoss()

#     # Training loop
#     for epoch in range(5):
#         model.train()
#         for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = loss_fn(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#     # Testing loop
#     model.eval()
#     test_loss = 0.0
#     total_recall = 0.0
#     total_accuracy = 0.0

#     with torch.no_grad():
#         for X_batch, y_batch in tqdm(test_loader, desc="Testing", leave=False):
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             outputs = model(X_batch)
#             test_loss += loss_fn(outputs, y_batch).item()
#             total_recall += recall(y_batch.cpu().numpy(), outputs.cpu().numpy())
#             total_accuracy += accuracy(y_batch.cpu().numpy(), outputs.cpu().numpy())

#     avg_recall = total_recall / len(test_loader)
#     avg_accuracy = total_accuracy / len(test_loader)

#     # Report metrics
#     session.report({
#         "loss": test_loss / len(test_loader),
#         "recall": avg_recall,
#         "accuracy": avg_accuracy
#     })

# from pathlib import Path

# if __name__ == "__main__":
#     ray.init(num_cpus=4)

#     # Gebruik een absoluut pad voor storage_path
#     storage_path = f"file://{Path('hypertuning_results').resolve()}"

#     config = {
#         "hidden_size": tune.randint(256, 512),
#         "dropout": tune.uniform(0.0, 0.2),
#         "output_size": 5,  # Multiclass classification with 5 classes
#     }

#     reporter = CLIReporter()
#     reporter.add_metric_column("Recall")
#     reporter.add_metric_column("Accuracy")

#     search = HyperOptSearch()
#     scheduler = AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         grace_period=1,
#         reduction_factor=3,
#         max_t=5,
#     )

#     analysis = tune.run(
#         train,
#         config=config,
#         metric="recall",
#         mode="max",
#         progress_reporter=reporter,
#         num_samples=5,  # Aantal experimenten aangepast
#         search_alg=search,
#         scheduler=scheduler,
#         verbose=1,
#         storage_path=storage_path,  # Gebruik absoluut pad
#     )

#     # Print de beste configuratie
#     print("Best configuration:", analysis.best_config)

#     # Zorg ervoor dat de map bestaat zonder URI
#     Path("hypertuning_results").mkdir(parents=True, exist_ok=True)

#     # Sla resultaten op als CSV
#     results_df = analysis.results_df
#     results_df.to_csv("hypertuning_results/results.csv", index=False)

#     ray.shutdown()



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
from src.datasets import HeartDataset2D
from src.metrics import Accuracy, Recall
from loguru import logger

logger.remove()
logger.add("logs/logfile.log", level="DEBUG")

# ResidualBlock for 2D Convolutions
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# CNN Model using ResidualBlock2D
class CNNModel2D(nn.Module):
    def __init__(self, config):
        super(CNNModel2D, self).__init__()
        self.blocks = nn.ModuleList()
        in_channels = 1

        for i in range(config["num_blocks"]):
            out_channels = config["hidden_size"] * (2 ** i)
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ) if i > 0 else None
            self.blocks.append(ResidualBlock2D(in_channels, out_channels, downsample=downsample, dropout=config["dropout"]))
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, config["output_size"])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

def train(config: Dict):
    # Gebruik het correcte pad naar de data-map
    data_dir = Path(__file__).parent.parent.resolve() / "data"  # Map buiten /dev/
    train_path = data_dir / "heart_big_train.parq"
    test_path = data_dir / "heart_big_test.parq"

    # Laad de data met HeartDataset2D
    train_dataset = HeartDataset2D(train_path, target="target", shape=(16, 12), apply_smote=True)
    test_dataset = HeartDataset2D(test_path, target="target", shape=(16, 12))

    # Gebruik een percentage van de data
    train_subset_size = int(0.5 * len(train_dataset))  # 50% van de trainingsdata
    test_subset_size = int(0.5 * len(test_dataset))    # 50% van de testdata

    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    recall = Recall(average="macro")
    accuracy = Accuracy()
    model = CNNModel2D(config)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Testing loop
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

    # Report metrics
    session.report({
        "loss": test_loss / len(test_loader),
        "recall": avg_recall,
        "accuracy": avg_accuracy
    })

if __name__ == "__main__":
    ray.init(num_cpus=4)

    # Gebruik een absoluut pad voor storage_path
    storage_path = f"file://{Path('hypertuning_results').resolve()}"

    config = {
        "hidden_size": tune.randint(16, 256),
        "dropout": tune.uniform(0.0, 0.2),
        "output_size": 5,  # Multiclass classification with 5 classes
        "num_blocks": tune.randint(1, 7),  # Kies tussen 1 en 6 blokken
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
        num_samples=10,  # Aantal experimenten aangepast
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
        storage_path=storage_path,  # Gebruik absoluut pad
    )

    # Print de beste configuratie
    print("Best configuration:", analysis.best_config)

    # Zorg ervoor dat de map bestaat zonder URI
    Path("hypertuning_results").mkdir(parents=True, exist_ok=True)

    # Sla resultaten op als CSV
    results_df = analysis.results_df
    results_df.to_csv("hypertuning_results/results.csv", index=False)

    ray.shutdown()
