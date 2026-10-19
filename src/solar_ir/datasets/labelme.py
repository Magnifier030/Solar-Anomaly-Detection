"""IR Labelme polygon dataset.

The loader only knows about images, Labelme polygons, and group-based splits. It
does not know which segmentation network will consume the tensors.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch.utils.data import Dataset


def _frames(image_dir: Path, label_dir: Path) -> list[str]:
    return sorted(
        path.stem
        for path in image_dir.glob("*.jpg")
        if (label_dir / f"{path.stem}.json").exists()
    )


def _has_hotspot(label_dir: Path, frame: str) -> bool:
    annotation = json.loads((label_dir / f"{frame}.json").read_text(encoding="utf-8"))
    return any(shape.get("label") == "hotspot" for shape in annotation.get("shapes", []))


def build_group_split(
    image_dir: Path,
    label_dir: Path,
    groups_csv: Path,
    validation_fraction: float = 0.2,
    split_strategy: str = "group",
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split frames by group, or stratified by frame for development only."""
    with groups_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    group_by_frame = {str(row["frame_id"]): str(row["group"]) for row in rows}
    frames = _frames(image_dir, label_dir)
    if split_strategy == "random_stratified":
        positive = [frame for frame in frames if _has_hotspot(label_dir, frame)]
        normal = [frame for frame in frames if frame not in set(positive)]
        generator = random.Random(seed)
        generator.shuffle(positive)
        generator.shuffle(normal)
        validation_count = max(1, round(len(frames) * validation_fraction))
        positive_count = round(len(positive) * validation_fraction)
        normal_count = validation_count - positive_count
        validation = positive[:positive_count] + normal[:normal_count]
        if len(validation) < validation_count:
            remaining = [frame for frame in frames if frame not in set(validation)]
            generator.shuffle(remaining)
            validation.extend(remaining[: validation_count - len(validation)])
        validation_set = set(validation)
        train_paths = [image_dir / f"{frame}.jpg" for frame in frames if frame not in validation_set]
        validation_paths = [image_dir / f"{frame}.jpg" for frame in frames if frame in validation_set]
        return train_paths, validation_paths
    if split_strategy != "group":
        raise ValueError("split_strategy must be group or random_stratified")
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
            positives += _has_hotspot(label_dir, frame)
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

    def __init__(
        self,
        image_paths: list[Path],
        label_dir: Path,
        image_size: int | tuple[int, int],
        *,
        augment: bool = False,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.label_dir = label_dir
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self.augment = augment
        self.augmentation = augmentation or {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        original_size = image.size
        mask_image = Image.new("L", original_size, 0)
        draw = ImageDraw.Draw(mask_image)
        annotation = json.loads((self.label_dir / f"{image_path.stem}.json").read_text(encoding="utf-8"))
        for shape in annotation.get("shapes", []):
            if shape.get("label") != "hotspot":
                continue
            points = [tuple(float(value) for value in point) for point in shape.get("points", [])]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        if self.augment:
            image, mask_image = self._training_transform(image, mask_image)
        else:
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            mask_image = mask_image.resize(self.image_size, Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.asarray(mask_image, dtype=np.float32) / 255.0).unsqueeze(0)
        return tensor, mask_tensor

    def _training_transform(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        import random

        if random.random() < float(self.augmentation.get("horizontal_flip_prob", 0.5)):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        crop_width, crop_height = self.image_size
        scale_min = float(self.augmentation.get("scale_min", 0.5))
        scale_max = float(self.augmentation.get("scale_max", 2.0))
        base_long_size = max(crop_width, crop_height)
        long_size = random.randint(
            max(1, int(base_long_size * scale_min)),
            max(1, int(base_long_size * scale_max)),
        )
        width, height = image.size
        if height > width:
            resized_height = long_size
            resized_width = max(1, int(width * long_size / height + 0.5))
        else:
            resized_width = long_size
            resized_height = max(1, int(height * long_size / width + 0.5))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)

        pad_width = max(0, crop_width - resized_width)
        pad_height = max(0, crop_height - resized_height)
        if pad_width or pad_height:
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=0)
        width, height = image.size
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        image = image.crop((left, top, left + crop_width, top + crop_height))
        mask = mask.crop((left, top, left + crop_width, top + crop_height))

        blur_probability = float(self.augmentation.get("gaussian_blur_prob", 0.5))
        if random.random() < blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, mask
