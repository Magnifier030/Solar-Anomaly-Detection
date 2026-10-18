"""Stable contracts between data, models, and the training task."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn


class SegmentationModelFactory(Protocol):
    def __call__(self, config: dict[str, Any]) -> nn.Module: ...


class MaskPostprocessor(Protocol):
    def __call__(self, mask: torch.Tensor, threshold: float) -> list[list[list[float]]]: ...
