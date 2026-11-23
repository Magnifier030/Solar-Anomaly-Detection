"""DINOv2-with-registers segmentation with a configurable number of input channels.

The patch projection is replaced so that it accepts ``channels`` planes. RGB planes
keep the pretrained weights; any extra plane (IR) starts from the mean RGB filter.
The encoder stays frozen except the patch projection, which must learn how to read
the new IR plane. A small convolutional head upsamples the 14-pixel token grid back to
input resolution. Outputs are logits for three independent targets.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoSegmentation(nn.Module):
    def __init__(
        self, backbone: nn.Module, channels: int, targets: int, head: str = "conv"
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.channels = channels
        self.num_registers = int(backbone.config.num_register_tokens)
        dim = int(backbone.config.hidden_size)
        if head == "linear":
            # Lowest-capacity head: one 1x1 projection of the frozen token grid.
            self.head = nn.Conv2d(dim, targets, 1)
        elif head != "conv":
            raise ValueError("model.head must be conv or linear")
        else:
            self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, targets, 1),
        )
        mean = torch.tensor(IMAGENET_MEAN)
        std = torch.tensor(IMAGENET_STD)
        if channels == 1:
            mean, std = mean.mean().view(1), std.mean().view(1)
        elif channels == 4:
            mean = torch.cat([mean, mean.mean().view(1)])
            std = torch.cat([std, std.mean().view(1)])
        self.register_buffer("mean", mean.view(1, channels, 1, 1))
        self.register_buffer("std", std.view(1, channels, 1, 1))

    def train(self, mode: bool = True) -> DinoSegmentation:
        super().train(mode)
        # Only the patch projection adapts; the transformer stays in eval mode.
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        size = images.shape[-2:]
        normalized = (images - self.mean) / self.std
        hidden = self.backbone(pixel_values=normalized).last_hidden_state
        tokens = hidden[:, 1 + self.num_registers :]
        side = int(tokens.shape[1] ** 0.5)
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
        logits = self.head(grid)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def _replace_patch_projection(backbone: nn.Module, channels: int) -> None:
    embeddings = backbone.embeddings.patch_embeddings
    old = embeddings.projection
    new = nn.Conv2d(channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride)
    with torch.no_grad():
        rgb_weight = old.weight
        if channels == 1:
            new.weight.copy_(rgb_weight.mean(dim=1, keepdim=True))
        elif channels == 3:
            new.weight.copy_(rgb_weight)
        elif channels == 4:
            new.weight.copy_(torch.cat([rgb_weight, rgb_weight.mean(dim=1, keepdim=True)], dim=1))
        else:
            raise ValueError("model.channels must be 1, 3 or 4")
        new.bias.copy_(old.bias)
    embeddings.projection = new
    embeddings.num_channels = channels
    backbone.config.num_channels = channels


def create_model(config: dict[str, Any]) -> DinoSegmentation:
    from transformers import Dinov2WithRegistersModel

    channels = int(config["channels"])
    targets = int(config.get("targets", 3))
    backbone = Dinov2WithRegistersModel.from_pretrained(
        str(config.get("encoder", "facebook/dinov2-with-registers-base"))
    )
    backbone.requires_grad_(False)
    _replace_patch_projection(backbone, channels)
    backbone.embeddings.patch_embeddings.projection.requires_grad_(True)
    head = str(config.get("head", "conv"))
    return DinoSegmentation(backbone, channels=channels, targets=targets, head=head)
