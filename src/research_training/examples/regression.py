"""Deterministic synthetic regression smoke task."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from research_training.task import TaskBundle


class RegressionTask:
    """A minimal plugin suitable for testing the full infrastructure."""

    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        task = config["task"]
        seed = int(config["training"]["seed"])
        data_generator = torch.Generator().manual_seed(seed)
        features = torch.randn(
            int(task.get("samples", 128)),
            int(task.get("features", 4)),
            generator=data_generator,
        )
        weights = torch.arange(1, features.shape[1] + 1, dtype=torch.float32)
        noise = torch.randn(features.shape[0], generator=data_generator)
        targets = features @ weights + 0.05 * noise
        split = int(0.8 * len(features))
        train = TensorDataset(features[:split], targets[:split, None])
        validation = TensorDataset(features[split:], targets[split:, None])
        loader_generator = torch.Generator().manual_seed(seed + 1)
        model = nn.Sequential(
            nn.Linear(features.shape[1], int(task.get("hidden_size", 16))), nn.ReLU(),
            nn.Linear(int(task.get("hidden_size", 16)), 1),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["training"]["learning_rate"])
        )
        return TaskBundle(
            model=model, optimizer=optimizer,
            train_loader=DataLoader(
                train,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=True,
                generator=loader_generator,
            ),
            validation_loader=DataLoader(validation, batch_size=64),
            metadata={
                "dataset_version": "synthetic_regression_v1",
                "train_samples": len(train),
                "validation_samples": len(validation),
            },
            train_generator=loader_generator,
        )

    def training_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features, targets = (value.to(device) for value in batch)
        loss = nn.functional.mse_loss(model(features), targets)
        return loss, {"rmse": float(torch.sqrt(loss.detach()))}

    def validation_metrics(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()
        errors: list[torch.Tensor] = []
        with torch.no_grad():
            for features, targets in loader:
                errors.append(model(features.to(device)) - targets.to(device))
        error = torch.cat(errors)
        model.train(was_training)
        return {"loss": float(torch.mean(error.square())), "mae": float(torch.mean(error.abs()))}


def create_task() -> RegressionTask:
    """Create the example plugin."""
    return RegressionTask()
