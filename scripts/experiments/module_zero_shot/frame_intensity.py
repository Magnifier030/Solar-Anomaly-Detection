"""Within-frame intensity outlier score per module, combined with the DINOv2 mutual score.

Texture features miss uniform brightness changes (e.g. a bypass-diode third of a module
turning bright), so each module also gets intensity statistics inside its polygon,
expressed relative to the other modules of the same frame (robust z, absolute value so
the score works for both white-hot and black-hot palettes). Rows follow the same order
as frame_mutual.py, so the saved DINO scores can be combined per module.

Usage: python frame_intensity.py ir|eo
"""

import sys
from collections import defaultdict
from pathlib import Path

import json

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
MODALITY = sys.argv[1]
ROOT = PROJECT / "data/raw/datasets/9_16_train_set/train" / ("IR" if MODALITY == "ir" else "EO")
ANOMALY = "hotspot" if MODALITY == "ir" else "crack"
STATS = ("mean", "p90", "p10", "std", "bright_frac")


def module_stats(gray, points):
    mask = Image.new("L", (gray.shape[1], gray.shape[0]), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in points], fill=1)
    values = gray[np.asarray(mask, bool)]
    if values.size < 20:
        values = np.zeros(20)
    return values


def robust_z(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median)) * 1.4826 + 1e-6
    return np.abs(x - median) / mad


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    return (ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")


def frame_rank(score, stems):
    """Rank inside each frame scaled to [0, 1] (1 = most anomalous)."""
    out = np.zeros_like(score)
    for stem in np.unique(stems):
        idx = np.where(stems == stem)[0]
        order = np.argsort(np.argsort(score[idx]))
        out[idx] = order / max(len(idx) - 1, 1)
    return out


def main():
    rows = []
    for path in sorted((ROOT / "label").glob("*.json")):
        shapes = [
            s for s in json.loads(path.read_text(encoding="utf-8"))["shapes"]
            if s["label"] in {"solar_panel", ANOMALY} and len(s["points"]) >= 3
        ]
        shapes = [
            s for s in shapes
            if np.ptp(np.array(s["points"])[:, 0]) >= 8 and np.ptp(np.array(s["points"])[:, 1]) >= 8
        ]
        if len(shapes) < 3:
            continue
        gray = np.asarray(Image.open(ROOT / "image" / (path.stem + ".jpg")).convert("L"), np.float32)
        frame_values = [module_stats(gray, s["points"]) for s in shapes]
        threshold = np.percentile(np.concatenate(frame_values), 90)
        feats = np.array([
            [v.mean(), np.percentile(v, 90), np.percentile(v, 10), v.std(), (v > threshold).mean()]
            for v in frame_values
        ])
        z = np.stack([robust_z(feats[:, j]) for j in range(feats.shape[1])], axis=1)
        for s, zz in zip(shapes, z):
            rows.append((path.stem, int(s["label"] == ANOMALY), zz))
    stems = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    Z = np.stack([r[2] for r in rows])
    dino = np.load(SCRATCH / f"mutual_{MODALITY}_224_k1.npz")
    assert (dino["stems"] == stems).all() and (dino["y"] == y).all(), "row order differs from frame_mutual"
    g = dino["g"]
    scores = {name: Z[:, j] for j, name in enumerate(STATS)}
    scores["intensity_max"] = Z.max(axis=1)
    scores["dino_k1"] = dino["raw"]
    scores["rank_avg(dino,intensity_max)"] = frame_rank(dino["raw"], stems) + frame_rank(Z.max(axis=1), stems)
    scores["rank_max(dino,intensity_max)"] = np.maximum(frame_rank(dino["raw"], stems), frame_rank(Z.max(axis=1), stems))
    groups = [grp for grp in sorted(set(g)) if y[g == grp].sum()]
    print(f"{MODALITY}: AUC by score (ALL | " + " ".join(groups) + ")")
    for name, s in scores.items():
        per = " ".join(f"{auc(s[g == grp], y[g == grp]):.3f}" for grp in groups)
        print(f"  {name:30s} {auc(s, y):.3f} | {per}")
    np.savez(SCRATCH / f"intensity_{MODALITY}.npz", stems=stems, y=y, g=g, Z=Z)


if __name__ == "__main__":
    main()
