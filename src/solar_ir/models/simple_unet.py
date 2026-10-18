"""Small dependency-free segmentation baseline used to validate the pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SimpleUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, base_channels, 3, padding=1), nn.ReLU(), ConvBlock(base_channels))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(), ConvBlock(base_channels * 2))
        self.middle = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU(), ConvBlock(base_channels * 4))
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.head = nn.Conv2d(base_channels, 1, 1)
        self.refine2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 1), nn.ReLU(), ConvBlock(base_channels * 2)
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1), nn.ReLU(), ConvBlock(base_channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.down1(value)
        second = self.down2(first)
        middle = self.middle(second)
        second = self.refine2(torch.cat([self.up2(middle), second], dim=1))
        first = self.refine1(torch.cat([self.up1(second), first], dim=1))
        return self.head(first)


def create_model(config: dict[str, Any]) -> nn.Module:
    return SimpleUNet(int(config.get("base_channels", 16)))
