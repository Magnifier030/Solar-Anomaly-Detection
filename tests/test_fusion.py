from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from solar_ir.datasets.paired import MODALITY_CHANNELS, PairedSolarDataset
from solar_ir.models.dino_seg import DinoSegmentation, _replace_patch_projection


def _write_pair(root: Path, stem: str) -> None:
    ir_image = root / "IR/image"
    ir_label = root / "IR/label"
    eo_image = root / "EO/image"
    eo_label = root / "EO/label"
    for folder in (ir_image, ir_label, eo_image, eo_label):
        folder.mkdir(parents=True, exist_ok=True)
    Image.new("L", (640, 512), 128).save(ir_image / f"{stem}.jpg")
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(eo_image / f"{stem}.jpg")
    # IR: a full-frame panel and a hotspot in the top-left quarter.
    ir_shapes = [
        {"label": "solar_panel", "points": [[0, 0], [640, 0], [640, 512], [0, 512]]},
        {"label": "hotspot", "points": [[0, 0], [160, 0], [160, 128], [0, 128]]},
    ]
    eo_shapes = [{"label": "crack", "points": [[1920, 1080], [960, 1080], [960, 540]]}]
    (ir_label / f"{stem}.json").write_text(json.dumps({"shapes": ir_shapes}), encoding="utf-8")
    (eo_label / f"{stem}.json").write_text(json.dumps({"shapes": eo_shapes}), encoding="utf-8")


@pytest.mark.parametrize("modalities", ["ir", "rgb", "rgbir"])
def test_channel_count_follows_modality(tmp_path: Path, modalities: str) -> None:
    _write_pair(tmp_path, "train_0001")
    dataset = PairedSolarDataset(
        [tmp_path / "IR/image/train_0001.jpg"],
        tmp_path / "IR/label",
        tmp_path / "EO/image",
        tmp_path / "EO/label",
        canvas=64,
        modalities=modalities,
    )
    image, target = dataset[0]
    assert image.shape == (MODALITY_CHANNELS[modalities], 64, 64)
    assert target.shape == (3, 64, 64)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_targets_come_from_their_own_modality(tmp_path: Path) -> None:
    _write_pair(tmp_path, "train_0001")
    dataset = PairedSolarDataset(
        [tmp_path / "IR/image/train_0001.jpg"],
        tmp_path / "IR/label",
        tmp_path / "EO/image",
        tmp_path / "EO/label",
        canvas=64,
        modalities="rgbir",
    )
    _, target = dataset[0]
    panel, hotspot, crack = (target[index].numpy() for index in range(3))
    assert panel.mean() == pytest.approx(1.0, abs=0.05)
    # Hotspot occupies the top-left quarter: 0.25 x 0.25 of the canvas.
    assert hotspot[:16, :16].mean() == pytest.approx(1.0)
    assert hotspot[32:, 32:].sum() == 0
    # Crack is a triangle in EO coordinates; it lies in the lower-right area only.
    assert crack[:20, :20].sum() == 0
    assert crack.sum() > 0


def test_rgb_planes_keep_pretrained_filters_and_ir_uses_their_mean() -> None:
    rgb_weight = torch.randn(8, 3, 14, 14)
    projection = nn.Conv2d(3, 8, 14, stride=14)
    with torch.no_grad():
        projection.weight.copy_(rgb_weight)
        projection.bias.zero_()
    backbone = SimpleNamespace(
        embeddings=SimpleNamespace(
            patch_embeddings=SimpleNamespace(projection=projection, num_channels=3)
        ),
        config=SimpleNamespace(num_channels=3),
    )
    _replace_patch_projection(backbone, 4)
    new = backbone.embeddings.patch_embeddings.projection
    assert new.in_channels == 4
    torch.testing.assert_close(new.weight[:, :3], rgb_weight)
    torch.testing.assert_close(new.weight[:, 3:], rgb_weight.mean(dim=1, keepdim=True))
    assert backbone.config.num_channels == 4


def test_segmentation_head_returns_full_resolution_logits() -> None:
    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_register_tokens=4, hidden_size=16)
            self.proj = nn.Conv2d(4, 16, 14, stride=14)

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            grid = self.proj(pixel_values).flatten(2).transpose(1, 2)
            registers = torch.zeros(grid.shape[0], 5, grid.shape[2])
            return SimpleNamespace(last_hidden_state=torch.cat([registers, grid], dim=1))

    model = DinoSegmentation(FakeBackbone(), channels=4, targets=3)
    images = torch.rand(2, 4, 56, 56)
    logits = model(images)
    assert logits.shape == (2, 3, 56, 56)
    assert np.isfinite(logits.detach().numpy()).all()
