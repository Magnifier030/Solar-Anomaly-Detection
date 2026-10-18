"""Stable boundary between research-specific code and the training engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaskBundle:
    """Everything the generic engine needs from one research task."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader[Any]
    validation_loader: DataLoader[Any]
    metadata: dict[str, Any]
    train_generator: torch.Generator | None = None


class ResearchTask(Protocol):
    """Implement this protocol for a new dataset/model/objective."""

    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        """Build fresh model, optimizer, and deterministic data loaders."""

    def training_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return differentiable loss and bounded scalar diagnostics."""

    def validation_metrics(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        """Return metrics including the configured selection metric."""
