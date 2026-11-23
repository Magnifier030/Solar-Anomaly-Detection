"""Reusable zero-shot within-frame module scoring (rectified DINOv2 mutual score + intensity std).

score_frame(image_rgb, polygons, k) -> dict of arrays per module:
  zr    robust z (within frame) of the DINOv2 mutual nearest-neighbour score
  zstd  |robust z| (within frame) of the rectified interior grey-level std
  fill  polygon area / minimum-area rectangle (low = cut by the frame border)
  score zr + zstd, with truncated modules (fill < 0.85) set to 0
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import Dinov2WithRegistersModel

from rectify import rectify

SIZE = 224
TOP = 0.10
FILL_MIN = 0.85
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base").cuda().eval()
    return _MODEL


@torch.no_grad()
def _tokens(crops):
    model = _model()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), 64):
        x = torch.from_numpy(np.stack(crops[i:i + 64])).cuda().permute(0, 3, 1, 2).float() / 255.0
        hidden = model(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + model.config.num_register_tokens:]
        out.append(torch.nn.functional.normalize(hidden, dim=-1))
    return torch.cat(out)


@torch.no_grad()
def _mutual(tokens, kth):
    m, p, d = tokens.shape
    k = max(1, int(round(p * TOP)))
    scores = torch.empty(m, device=tokens.device)
    for i in range(m):
        others = torch.cat([tokens[:i], tokens[i + 1:]])
        per_module = (2 - 2 * tokens[i] @ others.reshape(-1, d).T).view(p, m - 1, p).min(dim=2).values
        scores[i] = per_module.kthvalue(min(kth, m - 1), dim=1).values.topk(k).values.mean()
    return scores.cpu().numpy()


def robust_z(x, floor=0.0):
    """Robust z; the scale is at least floor*|median| so near-uniform frames cannot blow up."""
    median = np.median(x)
    scale = max(np.median(np.abs(x - median)) * 1.4826, floor * abs(median), 1e-6)
    return (x - median) / scale


def fill_ratio(points):
    polygon = np.asarray(points, np.float32)
    width, height = cv2.minAreaRect(polygon)[1]
    return cv2.contourArea(polygon) / max(width * height, 1e-6)


def score_frame(image_rgb, polygons, kth=1, use_std=True, floor=0.0):
    n = len(polygons)
    if n < 3:
        zeros = np.zeros(n)
        return {"zr": zeros, "zstd": zeros, "fill": np.ones(n), "score": zeros}
    grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    crops = [np.ascontiguousarray(rectify(image_rgb, np.asarray(p, float), SIZE, inset=0.03), np.uint8) for p in polygons]
    stds = np.array([rectify(grey, np.asarray(p, float), 64, inset=0.06).astype(np.float32).std() for p in polygons])
    fill = np.array([fill_ratio(p) for p in polygons])
    zr = robust_z(_mutual(_tokens(crops), kth), floor)
    zstd = np.abs(robust_z(stds, floor))
    # Intensity std helps IR (uniform bright bands) but not EO cracks (texture only).
    score = np.where(fill < FILL_MIN, 0.0, zr + zstd if use_std else zr)
    return {"zr": zr, "zstd": zstd, "fill": fill, "score": score}
