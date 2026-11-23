"""Optimizer policy shared by competition fine-tuning and external pretraining."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    optimizers = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "Adagrad": torch.optim.Adagrad,
    }
    name = str(training.get("optimizer", "AdamW"))
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer requires trainable parameters")
    optimizer = optimizers[name](parameters, lr=float(training.get("learning_rate", 1e-4)))
    schedule = str(training.get("scheduler", "none"))
    if schedule == "upstream_once":
        # Released DNA-Net code steps cosine once at initialization, not every epoch.
        initial_lr = float(training["learning_rate"])
        minimum_lr = float(training.get("min_lr", 1e-5))
        learning_rate = (
            minimum_lr
            + (initial_lr - minimum_lr) * (1 + math.cos(math.pi / int(training["max_epochs"]))) / 2
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    elif schedule != "none":
        raise ValueError("training.scheduler must be none or upstream_once")
    return optimizer
