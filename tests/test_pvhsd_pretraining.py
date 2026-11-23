from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from solar_ir.datasets.masks import ThermalMaskDataset
from solar_ir.models.dnanet import create_model
from solar_ir.preprocessing.pvhsd2025 import box_mask, family_split, file_hash
from solar_ir.pretraining import PVHSDPretrainingTask


def test_box_mask_clips_edges_and_preserves_empty_negatives():
    # Known normalized rectangle occupies exactly x=[0,2), y=[1,3).
    mask = np.asarray(box_mask((8, 4), [[0, 0.125, 0.5, 0.25, 0.5]]))
    assert mask.sum() == 4 * 255
    assert np.all(mask[1:3, :2] == 255)
    assert not np.asarray(box_mask((8, 4), [])).any()
    clipped = np.asarray(box_mask((8, 4), [[0, 0, 0, 0.25, 0.5]]))
    assert clipped[0, 0] == 255 and clipped.sum() == 255


def test_mask_pairs_use_one_channel_and_nearest_neighbor_targets(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    image = tmp_path / "panel.png"
    Image.new("L", (8, 4), 128).save(image)
    box_mask((8, 4), [[0, 0.5, 0.5, 0.5, 0.5]]).save(masks / image.name)
    dataset = ThermalMaskDataset([image], masks, (16, 8))
    x, y = dataset[0]
    assert x.shape == y.shape == (1, 8, 16)
    assert torch.allclose(x, torch.full_like(x, 128 / 255))
    assert set(y.unique().tolist()) == {0.0, 1.0}


def test_single_channel_checkpoint_load_is_strict(tmp_path: Path):
    model = create_model({"input_channels": 1})
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict()}, path)
    restored = create_model({"input_channels": 1, "pretrained_path": str(path)})
    torch.testing.assert_close(model.conv0_0[0].conv1.weight, restored.conv0_0[0].conv1.weight)
    with pytest.raises(RuntimeError):
        create_model({"input_channels": 3, "pretrained_path": str(path)})


def test_pretraining_detects_tampered_mask_and_overlapping_families(tmp_path: Path):
    for folder in ["images", "masks"]:
        (tmp_path / folder).mkdir()
    rows = []
    for i, split in enumerate(["train", "train", "validation"]):
        for folder in ["images", "masks"]:
            Image.new("L", (32, 32), 255).save(tmp_path / folder / f"{i}.png")
        rows.append(
            {
                "image": f"images/{i}.png",
                "mask": f"masks/{i}.png",
                "image_sha256": file_hash(tmp_path / "images" / f"{i}.png"),
                "mask_sha256": file_hash(tmp_path / "masks" / f"{i}.png"),
                "split": split,
                "family_hint": str(i),
            }
        )
    manifest = {
        "supervision": "bbox_raster_weak_mask",
        "input_channels": 1,
        "samples": rows,
        "signature": {"source_revision": "fixture"},
        "source_archive_sha256": "fixture",
        "split_policy": "fixture",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = {
        "task": {
            "dataset_root": str(tmp_path),
            "image_size": 32,
            "model": {"factory": "solar_ir.models.dnanet:create_model", "input_channels": 1},
        },
        "training": {"batch_size": 2, "loss": "soft_iou"},
    }
    task = PVHSDPretrainingTask()
    bundle = task.build(config, torch.device("cpu"))
    assert next(iter(bundle.train_loader))[0].shape == (2, 1, 32, 32)
    assert family_split("42", 7, 0.2) == family_split("42", 7, 0.2)
    Image.new("L", (32, 32), 0).save(tmp_path / "masks/0.png")
    with pytest.raises(ValueError, match="changed"):
        task.build(config, torch.device("cpu"))
    rows[2]["family_hint"] = rows[0]["family_hint"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        task.build(config, torch.device("cpu"))
