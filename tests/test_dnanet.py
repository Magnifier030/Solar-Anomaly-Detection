from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from solar_ir.datasets import build_group_split
from solar_ir.models.dnanet import create_model
from solar_ir.task import SolarIRTask


def test_official_soft_iou_uses_whole_batch_and_gradients():
    logits = torch.tensor(
        [[[[-2.0, 1.0], [0.0, 2.0]]], [[[3.0, -1.0], [1.0, 0.0]]]], requires_grad=True
    )
    masks = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]], [[[1.0, 1.0], [1.0, 0.0]]]])
    p = logits.sigmoid()
    intersection = (p * masks).sum()
    expected = 1 - (intersection + 1) / (p.sum() + masks.sum() - intersection + 1)
    loss = SolarIRTask._soft_iou_loss(logits, masks)
    torch.testing.assert_close(loss, expected)
    per_image = torch.stack(
        [SolarIRTask._soft_iou_loss(x, y) for x, y in zip(logits, masks, strict=True)]
    ).mean()
    assert not torch.isclose(loss, per_image)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_deep_supervision_averages_all_four_losses():
    class Outputs(nn.Module):
        def forward(self, x):
            return [x + offset for offset in (-2, -1, 0, 1)]

    task = SolarIRTask()
    task.loss_name = "soft_iou"
    x = torch.randn(2, 1, 4, 4, requires_grad=True)
    masks = torch.ones_like(x)
    model = Outputs()
    actual, _ = task.training_loss(model, (x, masks), torch.device("cpu"))
    expected = torch.stack([task._soft_iou_loss(y, masks) for y in model(x)]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()


def test_explicit_validation_groups_remain_disjoint(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    groups = tmp_path / "groups.csv"
    with groups.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_id", "group"])
        for i, group in enumerate(["A", "A", "B", "B", "C"]):
            frame = f"frame_{i}"
            (images / f"{frame}.jpg").touch()
            (labels / f"{frame}.json").write_text('{"shapes": []}')
            writer.writerow([frame, group])
    train, validation = build_group_split(images, labels, groups, validation_groups=["B"])
    assert {p.stem for p in validation} == {"frame_2", "frame_3"}
    assert not set(train) & set(validation)
    with pytest.raises(ValueError):
        build_group_split(images, labels, groups, validation_groups=["unknown"])


def test_official_checkpoint_strict_roundtrip(tmp_path: Path):
    model = create_model({})
    assert model.conv0_0[0].conv1.weight.shape == (16, 3, 3, 3)
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    restored = create_model({"pretrained_path": str(path)})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
