"""DNANet port based on the author's released architecture.

This module intentionally does not load checkpoints. The factory initializes a
fresh model so experiments are reproducible inside the common training engine.
The implementation keeps the nested dense paths and CBAM attention from the
original model, while exposing a normal ``create_model(config)`` plugin.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = self.fc2(self.relu(self.fc1(self.avg_pool(value))))
        maximum = self.fc2(self.relu(self.fc1(self.max_pool(value))))
        return self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = torch.mean(value, dim=1, keepdim=True)
        maximum, _ = torch.max(value, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        output = self.ca(output) * output
        output = self.sa(output) * output
        return self.relu(output + residual)


class DNANet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, filters: list[int], blocks: list[int], deep_supervision: bool) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        f = filters
        b = blocks
        layer = self._make_layer
        self.conv0_0 = layer(input_channels, f[0])
        self.conv1_0 = layer(f[0], f[1], b[0])
        self.conv2_0 = layer(f[1], f[2], b[1])
        self.conv3_0 = layer(f[2], f[3], b[2])
        self.conv4_0 = layer(f[3], f[4], b[3])
        self.conv0_1 = layer(f[0] + f[1], f[0])
        self.conv1_1 = layer(f[1] + f[2] + f[0], f[1], b[0])
        self.conv2_1 = layer(f[2] + f[3] + f[1], f[2], b[1])
        self.conv3_1 = layer(f[3] + f[4] + f[2], f[3], b[2])
        self.conv0_2 = layer(f[0] * 2 + f[1], f[0])
        self.conv1_2 = layer(f[1] * 2 + f[2] + f[0], f[1], b[0])
        self.conv2_2 = layer(f[2] * 2 + f[3] + f[1], f[2], b[1])
        self.conv0_3 = layer(f[0] * 3 + f[1], f[0])
        self.conv1_3 = layer(f[1] * 3 + f[2] + f[0], f[1], b[0])
        self.conv0_4 = layer(f[0] * 4 + f[1], f[0])
        self.conv0_4_final = layer(f[0] * 5, f[0])
        self.conv0_4_1x1 = nn.Conv2d(f[4], f[0], 1)
        self.conv0_3_1x1 = nn.Conv2d(f[3], f[0], 1)
        self.conv0_2_1x1 = nn.Conv2d(f[2], f[0], 1)
        self.conv0_1_1x1 = nn.Conv2d(f[1], f[0], 1)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.final1 = nn.Conv2d(f[0], num_classes, 1)
            self.final2 = nn.Conv2d(f[0], num_classes, 1)
            self.final3 = nn.Conv2d(f[0], num_classes, 1)
            self.final4 = nn.Conv2d(f[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(f[0], num_classes, 1)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, count: int = 1) -> nn.Sequential:
        layers = [ResCBAMBlock(in_channels, out_channels)]
        layers.extend(ResCBAMBlock(out_channels, out_channels) for _ in range(count - 1))
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self.conv0_0(value)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        final = self.conv0_4_final(torch.cat([
            self.up16(self.conv0_4_1x1(x4_0)), self.up8(self.conv0_3_1x1(x3_1)),
            self.up4(self.conv0_2_1x1(x2_2)), self.up(self.conv0_1_1x1(x1_3)), x0_4,
        ], 1))
        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(final)]
        return self.final(final)


def create_model(config: dict[str, Any]) -> nn.Module:
    backbone = str(config.get("backbone", "resnet_18"))
    blocks = {"resnet_10": [1, 1, 1, 1], "resnet_18": [2, 2, 2, 2], "resnet_34": [3, 4, 6, 3], "vgg_10": [1, 1, 1, 1]}.get(backbone)
    if blocks is None:
        raise ValueError(f"Unsupported DNANet backbone: {backbone}")
    channel_size = str(config.get("channel_size", "two"))
    filters = {"one": [4, 8, 16, 32, 64], "two": [8, 16, 32, 64, 128], "three": [16, 32, 64, 128, 256], "four": [32, 64, 128, 256, 512]}.get(channel_size)
    if filters is None:
        raise ValueError(f"Unsupported DNANet channel_size: {channel_size}")
    return DNANet(
        int(config.get("input_channels", 1)),
        int(config.get("num_classes", 1)),
        filters,
        blocks,
        bool(config.get("deep_supervision", True)),
    )
