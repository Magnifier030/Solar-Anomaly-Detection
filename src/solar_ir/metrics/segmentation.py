"""Binary segmentation metrics."""

from __future__ import annotations

import torch


def _counts(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float, float]:
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    intersection = float((prediction & truth).sum())
    union = float((prediction | truth).sum())
    total = float(prediction.sum() + truth.sum())
    return intersection, union, total


def mean_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    intersection, union, _ = _counts(logits, target)
    return (intersection + 1e-6) / (union + 1e-6)


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    intersection, _, total = _counts(logits, target)
    return (2 * intersection + 1e-6) / (total + 1e-6)
