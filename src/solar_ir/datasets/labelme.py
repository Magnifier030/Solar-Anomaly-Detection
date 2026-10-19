"""IR Labelme polygon dataset.

The loader only knows about images, Labelme polygons, and group-based splits. It
does not know which segmentation network will consume the tensors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def build_group_split(
    image_dir: Path,
    label_dir: Path,
    groups_csv: Path,
    validation_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Split frames by group, never by individual image."""
    with groups_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    group_by_frame = {str(row["frame_id"]): str(row["group"]) for row in rows}
    frames = sorted(path.stem for path in image_dir.glob("*.jpg") if (label_dir / f"{path.stem}.json").exists())
    groups = sorted({group_by_frame.get(frame, "ungrouped") for frame in frames})
    target = max(1, round(len(frames) * validation_fraction))
    group_frames = {
        group: [frame for frame in frames if group_by_frame.get(frame, "ungrouped") == group]
        for group in groups
    }
    positive_groups = []
    for group, group_frame_names in group_frames.items():
        positives = 0
        for frame in group_frame_names:
            annotation = json.loads((label_dir / f"{frame}.json").read_text(encoding="utf-8"))
            positives += any(shape.get("label") != "solar_panel" for shape in annotation.get("shapes", []))
        if positives:
            positive_groups.append(group)

    # Keep the split group-disjoint, but do not accidentally create a
    # validation set containing only normal frames. The closest positive group
    # to the requested size is a deterministic and leakage-safe baseline.
    candidates = positive_groups or groups
    validation_groups: set[str] = {
        min(candidates, key=lambda group: (abs(len(group_frames[group]) - target), group))
    }
    count = len(group_frames[next(iter(validation_groups))])
    for group in sorted(groups, key=lambda item: (abs(len(group_frames[item]) - max(0, target - count)), item)):
        if group in validation_groups:
            continue
        if count >= target:
            break
        validation_groups.add(group)
        count += len(group_frames[group])
    train = [image_dir / f"{frame}.jpg" for frame in frames if group_by_frame.get(frame, "ungrouped") not in validation_groups]
    validation = [image_dir / f"{frame}.jpg" for frame in frames if group_by_frame.get(frame, "ungrouped") in validation_groups]
    if not train or not validation:
        raise ValueError("Group split produced an empty train or validation split")
    return train, validation


class ThermalPolygonDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return one-channel IR tensor and a binary hotspot mask."""

    def __init__(self, image_paths: list[Path], label_dir: Path, image_size: int) -> None:
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        original_size = image.size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)

        mask_image = Image.new("L", original_size, 0)
        draw = ImageDraw.Draw(mask_image)
        annotation = json.loads((self.label_dir / f"{image_path.stem}.json").read_text(encoding="utf-8"))
        for shape in annotation.get("shapes", []):
            if shape.get("label") != "hotspot":
                continue
            points = [tuple(float(value) for value in point) for point in shape.get("points", [])]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        mask = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        return tensor, mask_tensor
