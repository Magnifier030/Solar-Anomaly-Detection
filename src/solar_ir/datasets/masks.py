"""Single-channel image and binary PNG mask pairs with synchronized transforms."""

from __future__ import annotations

import torch
from PIL import Image

from solar_ir.datasets.labelme import ThermalPolygonDataset


class ThermalMaskDataset(ThermalPolygonDataset):
    """Use label_dir for PNG masks; share transforms with the polygon dataset."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        with Image.open(path) as source:
            image = source.convert("L")
        with Image.open(self.label_dir / f"{path.stem}.png") as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {path}")
        return self._prepare_tensors(image, mask)
