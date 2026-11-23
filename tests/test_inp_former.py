from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from solar_ir.models.inp_former import InpFormer


class FakeViT(nn.Module):
    """Patch-14 stand-in for DINOv2 that exposes the same intermediate-layer API."""

    def __init__(self, dim: int = 32, depth: int = 10) -> None:
        super().__init__()
        self.embed_dim = dim
        self.depth = depth
        self.patchify = nn.Conv2d(3, dim, 14, stride=14)
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))

    def get_intermediate_layers(
        self, images: torch.Tensor, n: list[int]
    ) -> tuple[torch.Tensor, ...]:
        tokens = self.patchify(images).flatten(2).transpose(1, 2)
        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = torch.tanh(block(tokens))
            if index in n:
                outputs.append(tokens)
        return tuple(outputs)


def make_model(**overrides: float) -> InpFormer:
    settings = {
        "encoder_size": 28,
        "encoder_layers": [[2, 3], [6, 7]],
        "num_prototypes": 3,
        "decoder_depth": 4,
        "heads": 4,
        "gamma": 3.0,
        "coherence_weight": 0.2,
        "score_scale": 20.0,
        "score_offset": 0.3,
        "blur_sigma": 1.0,
    }
    settings.update(overrides)
    return InpFormer(encoder=FakeViT(), **settings)


def test_forward_returns_logit_map_at_input_resolution():
    model = make_model().eval()
    images = torch.rand(2, 1, 40, 56)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 1, 40, 56)
    assert torch.isfinite(logits).all()


def test_objective_trains_only_prototypes_and_decoder():
    model = make_model().train()
    loss, diagnostics = model.objective(torch.rand(2, 1, 28, 28))
    assert set(diagnostics) == {"soft_mining_loss", "coherence_loss"}
    loss.backward()
    assert model.extractor.tokens.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_soft_mining_keeps_cosine_value_but_scales_gradient():
    model = make_model(gamma=0.0).train()
    group = torch.randn(2, 4, 8)
    recon = torch.randn(2, 4, 8, requires_grad=True)
    loss = model._soft_mining_loss([group], [recon])
    expected = (1.0 - F.cosine_similarity(group, recon, dim=-1)).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(recon.grad).all()


def test_coherence_loss_is_zero_when_every_patch_matches_a_prototype():
    features = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    prototypes = torch.tensor([[[3.0, 0.0], [0.0, 1.0]]])
    assert InpFormer._coherence_loss(features, prototypes).item() < 1e-6
