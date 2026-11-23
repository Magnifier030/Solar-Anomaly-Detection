"""Paired IR/EO dataset for three-target solar defect segmentation.

Each frame has an IR image and an EO (RGB) image taken at the same time. The two
modalities are not spatially registered: EO is 1920×1080 and shows a different field
of view, so both images are resized independently onto a shared square canvas and
stacked channel-wise. Targets are rasterized per modality on that canvas:
``solar_panel`` and ``hotspot`` come from the IR annotation, ``crack`` from the EO
annotation. Each target is a separate binary channel because the classes overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

TARGET_CLASSES = ("solar_panel", "hotspot", "crack")
MODALITY_CHANNELS = {"ir": 1, "rgb": 3, "rgbir": 4}


def _boundary(label_path: Path, source_size: tuple[int, int], canvas: int) -> np.ndarray:
    """Outline pixels of every module polygon (solar_panel or hotspot), scaled to canvas."""
    mask = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(mask)
    scale_x = canvas / source_size[0]
    scale_y = canvas / source_size[1]
    annotation = json.loads(label_path.read_text(encoding="utf-8"))
    for shape in annotation.get("shapes", []):
        if shape.get("label") not in {"solar_panel", "hotspot"}:
            continue
        points = [(float(x) * scale_x, float(y) * scale_y) for x, y in shape.get("points", [])]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=255, width=3)
    return np.asarray(mask, dtype=np.float32) / 255.0


def _rasterize(
    label_path: Path, class_name: str, source_size: tuple[int, int], canvas: int
) -> np.ndarray:
    """Binary mask for one class, with polygons scaled from source to canvas pixels."""
    mask = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(mask)
    scale_x = canvas / source_size[0]
    scale_y = canvas / source_size[1]
    annotation = json.loads(label_path.read_text(encoding="utf-8"))
    for shape in annotation.get("shapes", []):
        if shape.get("label") != class_name:
            continue
        points = [(float(x) * scale_x, float(y) * scale_y) for x, y in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


class PairedSolarDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return ``(image [C,S,S] in [0,1], target [3,S,S])`` for one frame."""

    def __init__(
        self,
        frame_paths: list[Path],
        ir_label_dir: Path,
        eo_image_dir: Path,
        eo_label_dir: Path,
        canvas: int,
        modalities: str,
        normalize: str = "none",
        photometric: bool = False,
        invert_probability: float = 0.0,
        target_dir: Path | None = None,
        module_boundary: bool = False,
    ) -> None:
        if modalities not in MODALITY_CHANNELS:
            raise ValueError("modalities must be ir, rgb or rgbir")
        if normalize not in {"none", "percentile"}:
            raise ValueError("normalize must be none or percentile")
        self.frame_paths = frame_paths
        self.ir_label_dir = ir_label_dir
        self.eo_image_dir = eo_image_dir
        self.eo_label_dir = eo_label_dir
        self.canvas = canvas
        self.modalities = modalities
        self.normalize = normalize
        self.photometric = photometric
        self.invert_probability = invert_probability
        # Pseudo-label self-training: targets come from <target_dir>/<frame>.npy instead.
        self.target_dir = target_dir
        self.module_boundary = module_boundary

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ir_path = self.frame_paths[index]
        stem = ir_path.stem
        eo_path = self.eo_image_dir / f"{stem}.jpg"
        ir_image = Image.open(ir_path).convert("L")
        eo_image = Image.open(eo_path).convert("RGB")
        ir_size = ir_image.size
        eo_size = eo_image.size

        planes: list[np.ndarray] = []
        if self.modalities in {"rgb", "rgbir"}:
            resized = (
                np.asarray(
                    eo_image.resize((self.canvas, self.canvas), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
            planes.extend(resized[:, :, channel] for channel in range(3))
        if self.modalities in {"ir", "rgbir"}:
            resized = np.asarray(
                ir_image.resize((self.canvas, self.canvas), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            if self.normalize == "percentile":
                # Per-frame stretch: groups differ in overall brightness (mean 21 to 117).
                low, high = np.percentile(resized, [1, 99])
                resized = np.clip((resized - low) / max(high - low, 1.0), 0.0, 1.0)
            else:
                resized = resized / 255.0
            if self.invert_probability and np.random.random() < self.invert_probability:
                # Groups use opposite palettes: hotspots are darker than the panel in some
                # groups and brighter in others. Inversion makes the model read contrast.
                resized = 1.0 - resized
            if self.photometric:
                # Random gamma, contrast and brightness: day/night and sensor-gain shifts.
                resized = np.clip(resized, 0.0, 1.0) ** np.random.uniform(0.6, 1.6)
                mean = resized.mean()
                resized = (resized - mean) * np.random.uniform(0.7, 1.3) + mean
                resized = np.clip(resized + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
            planes.append(resized)
        image = torch.from_numpy(np.stack(planes, axis=0))

        if self.target_dir is not None:
            pseudo = np.load(self.target_dir / f"{stem}.npy").astype(np.float32)
            return image, torch.from_numpy(pseudo)
        planes_target = [
            _rasterize(self.ir_label_dir / f"{stem}.json", "solar_panel", ir_size, self.canvas),
            _rasterize(self.ir_label_dir / f"{stem}.json", "hotspot", ir_size, self.canvas),
            _rasterize(self.eo_label_dir / f"{stem}.json", "crack", eo_size, self.canvas),
        ]
        if self.module_boundary:
            # Module outlines from IR polygons (panel and hotspot modules), for instance split.
            planes_target.append(
                _boundary(self.ir_label_dir / f"{stem}.json", ir_size, self.canvas)
            )
        target = np.stack(planes_target, axis=0)
        return image, torch.from_numpy(target)


def channel_count(modalities: str) -> int:
    return MODALITY_CHANNELS[modalities]
