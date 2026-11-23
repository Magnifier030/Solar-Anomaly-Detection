from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from solar_ir.datasets.labelme import build_group_split
from solar_ir.datasets.splits import photo_stratified_split


def test_photo_split_preserves_scene_label_coverage_and_reproducibility():
    groups = {f"{g}_{label}_{i}": g for g in ["G01", "G02"] for label in [0, 1] for i in range(10)}
    positive = {frame: frame.split("_")[1] == "1" for frame in groups}
    frames = list(groups)
    train, valid = photo_stratified_split(frames, groups, positive, 0.2, 7)
    assert len(valid) == 8
    assert set(train).isdisjoint(valid)
    assert set(train) | set(valid) == set(frames)
    for group in ["G01", "G02"]:
        for label in [False, True]:
            assert sum(groups[f] == group and positive[f] == label for f in valid) == 2
    assert (train, valid) == photo_stratified_split(frames[::-1], groups, positive, 0.2, 7)
    assert valid != photo_stratified_split(frames, groups, positive, 0.2, 8)[1]


def test_photo_split_singletons_and_invalid_inputs():
    groups = {"single": "G01", "a": "G02", "b": "G02"}
    positives = dict.fromkeys(groups, True)
    train, valid = photo_stratified_split(list(groups), groups, positives, 0.2, 7)
    assert "single" in train and len(valid) == 1
    with pytest.raises(ValueError, match="unique"):
        photo_stratified_split(["a", "a"], groups, positives, 0.2, 7)
    with pytest.raises(ValueError, match="empty"):
        photo_stratified_split(["single"], groups, positives, 0.2, 7)


def test_photo_split_loader_dispatch_and_group_config_guard(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    csv_path = tmp_path / "groups.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_id", "group"])
        writer.writeheader()
        for i in range(10):
            frame = f"frame{i}"
            (images / f"{frame}.jpg").touch()
            (labels / f"{frame}.json").write_text(json.dumps({"shapes": [{"label": "hotspot"}]}))
            writer.writerow({"frame_id": frame, "group": "G02"})
    train, valid = build_group_split(images, labels, csv_path, split_strategy="photo_stratified")
    assert len(train) == 8 and len(valid) == 2
    with pytest.raises(ValueError, match="validation_groups"):
        build_group_split(
            images, labels, csv_path, split_strategy="photo_stratified", validation_groups=["G02"]
        )


def test_photo_config_changes_only_split_and_run_identity():
    root = Path(__file__).resolve().parents[1] / "configs/pvhsd2025"
    original = yaml.safe_load((root / "finetune_solar_dnanet_1ch_cudnn.yaml").read_text())
    photo = yaml.safe_load((root / "finetune_solar_dnanet_1ch_cudnn_photo.yaml").read_text())
    assert photo["name"] != original["name"]
    assert photo["task"]["split_strategy"] == "photo_stratified"
    assert "validation_groups" not in photo["task"]
    photo["name"] = original["name"]
    photo["task"]["split_strategy"] = original["task"]["split_strategy"]
    photo["task"]["validation_groups"] = original["task"]["validation_groups"]
    assert photo == original
