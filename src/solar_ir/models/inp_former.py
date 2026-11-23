"""INP-Former: intrinsic normal prototypes from a single image (Luo et al., CVPR 2025).

The frozen DINOv2-with-registers ViT-B/14 encodes the image. Learnable prototype
queries attend to the image's own patch tokens to form Intrinsic Normal Prototypes
(INPs). An INP-guided decoder reconstructs each encoder group using only ReLU
attention over those INPs, so regions that cannot be expressed by normal prototypes
reconstruct poorly. The regional cosine distance between encoder and decoder features
is the anomaly map.

Training is unsupervised: ``InpFormer.objective`` uses only the soft mining loss and
the INP coherence loss, so it never reads a mask. The output of ``forward`` is a
logit map, ``score_scale * (anomaly - score_offset)``, so the segmentation metrics and
checkpoint selection in ``solar_ir.task`` work unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.fc2 = nn.Linear(dim * ratio, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class INPExtractor(nn.Module):
    """Learnable prototype queries cross-attend to the image's normal tokens."""

    def __init__(self, dim: int, num_prototypes: int, heads: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_prototypes, dim) * 0.02)
        self.norm_query = nn.LayerNorm(dim)
        self.norm_key = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = self.tokens.expand(features.shape[0], -1, -1)
        key_value = self.norm_key(features)
        attended = query + self.attention(self.norm_query(query), key_value, key_value)[0]
        return attended + self.ffn(self.norm_ffn(attended))


class INPGuidedLayer(nn.Module):
    """Queries from decoder tokens; keys and values from INPs; ReLU instead of softmax."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(dim)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_prototypes = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim)

    def forward(self, tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        prototypes = self.norm_prototypes(prototypes)
        attention = F.relu(
            self.query(self.norm_tokens(tokens)) @ self.key(prototypes).transpose(1, 2)
        )
        # The first residual is intentionally absent: it would carry anomalous
        # features forward into the reconstruction.
        attended = (attention * self.scale) @ self.value(prototypes)
        return attended + self.ffn(self.norm_ffn(attended))


class InpFormer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_size: int,
        encoder_layers: list[list[int]],
        num_prototypes: int,
        decoder_depth: int,
        heads: int,
        gamma: float,
        coherence_weight: float,
        score_scale: float,
        score_offset: float,
        blur_sigma: float,
    ) -> None:
        super().__init__()
        if encoder_size % 14 != 0:
            raise ValueError("model.encoder_size must be a multiple of 14")
        self.encoder = encoder.eval().requires_grad_(False)
        self.encoder_size = encoder_size
        self.encoder_layers = encoder_layers
        self.gamma = gamma
        self.coherence_weight = coherence_weight
        self.score_scale = score_scale
        self.score_offset = score_offset
        self.blur_sigma = blur_sigma
        dim = int(encoder.embed_dim)
        self.extractor = INPExtractor(dim, num_prototypes, heads)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim * len(encoder_layers), dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.decoder = nn.ModuleList(INPGuidedLayer(dim) for _ in range(decoder_depth))
        # Group l is reconstructed at decoder depth (l + 1) * depth / L.
        self.taps = [
            (index + 1) * decoder_depth // len(encoder_layers) - 1
            for index in range(len(encoder_layers))
        ]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> InpFormer:
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return one [B, N, C] feature per encoder group, summed over its layers."""
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        images = F.interpolate(images, size=(self.encoder_size, self.encoder_size), mode="bilinear")
        images = (images - self.mean) / self.std
        flat_layers = sorted({layer for group in self.encoder_layers for layer in group})
        with torch.no_grad():
            tokens = self.encoder.get_intermediate_layers(images, n=flat_layers)
        by_layer = dict(zip(flat_layers, tokens, strict=True))
        return [sum(by_layer[layer] for layer in group) for group in self.encoder_layers]

    def _reconstruct(self, groups: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        prototypes = self.extractor(sum(groups))
        tokens = self.bottleneck(torch.cat(groups, dim=-1))
        reconstructions: list[torch.Tensor] = []
        for depth, layer in enumerate(self.decoder):
            tokens = layer(tokens, prototypes)
            if depth in self.taps:
                reconstructions.append(tokens)
        return prototypes, reconstructions

    @staticmethod
    def _regional_distance(groups: list[torch.Tensor], recons: list[torch.Tensor]) -> torch.Tensor:
        """Per-position cosine distance for each group, shape [L, B, N]."""
        return torch.stack(
            [1.0 - F.cosine_similarity(g, r, dim=-1) for g, r in zip(groups, recons, strict=True)]
        )

    def _gaussian_blur(self, value: torch.Tensor) -> torch.Tensor:
        radius = int(4 * self.blur_sigma)
        offsets = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
        kernel = torch.exp(-(offsets**2) / (2 * self.blur_sigma**2))
        kernel = kernel / kernel.sum()
        value = F.conv2d(value, kernel.view(1, 1, 1, -1), padding=(0, radius))
        return F.conv2d(value, kernel.view(1, 1, -1, 1), padding=(radius, 0))

    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        groups = self._encode(images)
        _, recons = self._reconstruct(groups)
        distance = self._regional_distance(groups, recons).mean(dim=0)
        side = int(math.sqrt(distance.shape[1]))
        amap = distance.view(distance.shape[0], 1, side, side)
        amap = F.interpolate(amap, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self._gaussian_blur(amap)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        amap = self.anomaly_map(images)
        return self.score_scale * (amap - self.score_offset)

    def objective(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Paper losses on normal images: soft mining loss plus INP coherence loss."""
        groups = self._encode(images)
        prototypes, recons = self._reconstruct(groups)
        soft_mining = self._soft_mining_loss(groups, recons)
        coherence = self._coherence_loss(sum(groups), prototypes)
        loss = soft_mining + self.coherence_weight * coherence
        return loss, {
            "soft_mining_loss": float(soft_mining.detach()),
            "coherence_loss": float(coherence.detach()),
        }

    def _soft_mining_loss(
        self, groups: list[torch.Tensor], recons: list[torch.Tensor]
    ) -> torch.Tensor:
        """Cosine loss whose gradients are reweighted toward hard positions.

        The forward value is unchanged; the gradient flowing into each decoder token is
        scaled by ``w = (d / mean(d)) ** gamma``, following the paper's gradient adjustment.
        """
        losses = []
        for group, recon in zip(groups, recons, strict=True):
            with torch.no_grad():
                distance = 1.0 - F.cosine_similarity(group, recon, dim=-1)
                weight = (distance / distance.mean().clamp_min(1e-6)) ** self.gamma
            adjusted = weight.unsqueeze(-1) * recon + (1 - weight.unsqueeze(-1)) * recon.detach()
            losses.append((1.0 - F.cosine_similarity(group, adjusted, dim=-1)).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def _coherence_loss(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Mean cosine distance from each patch to its nearest INP."""
        similarity = F.normalize(features, dim=-1) @ F.normalize(prototypes, dim=-1).transpose(1, 2)
        return (1.0 - similarity.max(dim=-1).values).mean()


class HuggingFaceDinoV2(nn.Module):
    """Adapts a Hugging Face DINOv2-with-registers model to ``get_intermediate_layers``.

    Only the architecture code of the installed ``transformers`` package runs; the
    repository itself supplies weights and configuration, never Python code.
    """

    def __init__(self, model: nn.Module, num_registers: int) -> None:
        super().__init__()
        self.model = model
        self.num_registers = num_registers
        self.embed_dim = int(model.config.hidden_size)

    def get_intermediate_layers(
        self, images: torch.Tensor, n: list[int]
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.model(pixel_values=images, output_hidden_states=True).hidden_states
        # hidden_states[0] is the embedding; block i outputs hidden_states[i + 1].
        # Drop [CLS] and register tokens, keeping only the patch tokens.
        start = 1 + self.num_registers
        return tuple(self.model.layernorm(hidden_states[layer + 1][:, start:]) for layer in n)


def create_model(config: dict[str, Any]) -> InpFormer:
    from transformers import Dinov2WithRegistersModel

    repository = str(config.get("encoder", "facebook/dinov2-with-registers-base"))
    backbone = Dinov2WithRegistersModel.from_pretrained(repository)
    encoder = HuggingFaceDinoV2(backbone, num_registers=int(backbone.config.num_register_tokens))
    layers = config.get("encoder_layers", [[2, 3, 4, 5], [6, 7, 8, 9]])
    return InpFormer(
        encoder=encoder,
        encoder_size=int(config.get("encoder_size", 224)),
        encoder_layers=[[int(layer) for layer in group] for group in layers],
        num_prototypes=int(config.get("num_prototypes", 6)),
        decoder_depth=int(config.get("decoder_depth", 8)),
        heads=int(config.get("heads", 8)),
        gamma=float(config.get("gamma", 3.0)),
        coherence_weight=float(config.get("coherence_weight", 0.2)),
        score_scale=float(config.get("score_scale", 20.0)),
        score_offset=float(config.get("score_offset", 0.3)),
        blur_sigma=float(config.get("blur_sigma", 4.0)),
    )
