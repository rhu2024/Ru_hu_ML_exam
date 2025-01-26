from pathlib import Path
from typing import Dict

import numpy as np
import ray
import torch
from filelock import FileLock
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from src.datasets import HeartDataset1D
from src.metrics import Recall

logger.remove()
logger.add("logs/logfile.log", level="DEBUG")


def train_heart_model(config: Dict):
    train_file = Path("data/train.parquet")
    test_file = Path("data/test.parquet")

    # Controleer of de Parquet-bestanden bestaan
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file.resolve()}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file.resolve()}")

    # Load datasets with and without SMOTE
    with FileLock(train_file.parent / ".lock"):
        train_dataset = HeartDataset1D(train_file, target="target", apply_smote=True)
        test_dataset = HeartDataset1D(test_file, target="target", apply_smote=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False
    )

    # Define model
    input_size = train_dataset.x.shape[1]  # Dynamisch afgeleid van de dataset
    model = CNNWithResiduals(
        input_shape=(64, 64),  # Pas aan voor 2D als nodig
        num_classes=2,
        num_blocks=config["num_blocks"],
        dropout_rate=config["dropout_rate"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set up optimizer, loss, and metrics
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    metric = Recall(average="macro")

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    recall = metric(all_targets, all_outputs)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(model.state_dict(), path)
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report(recall=recall, checkpoint=checkpoint)


if __name__ == "__main__":
    ray.init(num_cpus=4)

    config = {
        "num_blocks": tune.randint(1, 5),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 5,
    }

    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=3)
    search = HyperOptSearch()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_heart_model),
            resources={"cpu": 2, "gpu": 0},
        ),
        tune_config=tune.TuneConfig(
            metric="recall",
            mode="max",
            scheduler=scheduler,
            num_samples=10,
            search_alg=search,
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("recall", "max")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final recall: {}".format(best_result.metrics["recall"]))

    ray.shutdown()
