"""Zero-shot within-frame mutual scoring of module crops (MuSc-style, no training).

Modules in one frame share panel type, lighting and IR palette, and anomalies are rare.
Each module's DINOv2 patch tokens are compared with the patch tokens of all OTHER
modules in the same frame; a patch's score is its nearest-neighbour distance, and the
module score is the mean of its top 10% patch scores. No labels are used for scoring,
so AUC is reported on every group.

Usage: python frame_mutual.py ir|eo [crop_size]
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Dinov2WithRegistersModel

from rectify import rectify

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
MODALITY = sys.argv[1]
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 224
ROOT = PROJECT / "data/raw/datasets/9_16_train_set/train" / ("IR" if MODALITY == "ir" else "EO")
ANOMALY = "hotspot" if MODALITY == "ir" else "crack"
GROUPS_CSV = PROJECT / "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
TOP = 0.10

groups = {
    r["frame_id"]: r["group"]
    for r in csv.DictReader(open(GROUPS_CSV, encoding="utf-8", newline=""))
}
model = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base").cuda().eval()
REGISTERS = model.config.num_register_tokens


RECTIFY = len(sys.argv) > 4 and sys.argv[4] == "rect"


def crops(label_path):
    image = Image.open(ROOT / "image" / (label_path.stem + ".jpg")).convert("RGB")
    array = np.asarray(image)
    gray = np.asarray(image.convert("L"))
    out = []
    for shape in json.loads(label_path.read_text(encoding="utf-8"))["shapes"]:
        if shape["label"] not in {"solar_panel", ANOMALY} or len(shape["points"]) < 3:
            continue
        points = np.array(shape["points"], float)
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        # Fill ratio: polygon area over its minimum-area rectangle; low for modules cut by
        # the frame border, which are geometric outliers rather than defects.
        polygon = points.astype(np.float32)
        box_w, box_h = cv2.minAreaRect(polygon)[1]
        fill = cv2.contourArea(polygon) / max(box_w * box_h, 1e-6)
        if RECTIFY:
            crop = rectify(array, points, SIZE, inset=0.03)
            interior = rectify(gray, points, 64, inset=0.06).astype(np.float32)
            stats = (interior.std(), fill)
        else:
            crop = np.asarray(
                image.crop((int(x0), int(y0), int(x1), int(y1))).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            )
            stats = (0.0, fill)
        out.append((np.ascontiguousarray(crop, np.uint8), int(shape["label"] == ANOMALY), stats))
    return out


@torch.no_grad()
def patch_tokens(batch):
    tokens = []
    for i in range(0, len(batch), 64):
        x = torch.from_numpy(np.stack(batch[i:i + 64])).cuda().permute(0, 3, 1, 2).float() / 255.0
        hidden = model(pixel_values=(x - MEAN) / STD).last_hidden_state[:, 1 + REGISTERS:]
        tokens.append(torch.nn.functional.normalize(hidden, dim=-1))
    return torch.cat(tokens)  # [M, P, D]


KTH = int(sys.argv[3]) if len(sys.argv) > 3 else 1


@torch.no_grad()
def mutual_scores(tokens):
    m, p, d = tokens.shape
    scores = torch.empty(m, device=tokens.device)
    k = max(1, int(round(p * TOP)))
    for i in range(m):
        query = tokens[i]  # [P, D]
        others = torch.cat([tokens[:i], tokens[i + 1:]])  # [M-1, P, D]
        # Nearest patch inside each other module: [P, M-1].
        per_module = (2 - 2 * query @ others.reshape(-1, d).T).view(p, m - 1, p).min(dim=2).values
        # k-th closest module, so one look-alike anomalous module cannot mask an anomaly.
        kth = min(KTH, m - 1)
        nearest = per_module.kthvalue(kth, dim=1).values
        scores[i] = nearest.topk(k).values.mean()
    return scores.cpu().numpy()


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    return (ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")


def robust_z(x):
    median = np.median(x)
    return (x - median) / (np.median(np.abs(x - median)) * 1.4826 + 1e-6)


def main():
    rows = []  # (stem, group, label, raw, zscore, robust z of raw, |robust z| of interior std)
    paths = sorted((ROOT / "label").glob("*.json"))
    for index, path in enumerate(paths, 1):
        items = crops(path)
        if len(items) < 3:
            continue
        labels = np.array([label for _, label, _ in items])
        raw = mutual_scores(patch_tokens([crop for crop, _, _ in items]))
        z = (raw - np.median(raw)) / (raw.std() + 1e-6)
        zr = robust_z(raw)
        zstd = np.abs(robust_z(np.array([stats[0] for _, _, stats in items])))
        fills = [stats[1] for _, _, stats in items]
        for r, zz, a, b, f, label in zip(raw, z, zr, zstd, fills, labels):
            rows.append((path.stem, groups.get(path.stem, "?"), int(label), float(r), float(zz), float(a), float(b), float(f)))
        if index % 100 == 0:
            print(f"frames {index}/{len(paths)}", flush=True)
    stems = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    raw = np.array([r[3] for r in rows])
    z = np.array([r[4] for r in rows])
    zr = np.array([r[5] for r in rows])
    zstd = np.array([r[6] for r in rows])
    fill = np.array([r[7] for r in rows])
    combined = zr + zstd
    suffix = "_rect" if RECTIFY else ""
    np.savez(
        SCRATCH / f"mutual_{MODALITY}_{SIZE}_k{KTH}{suffix}.npz",
        stems=stems, g=g, y=y, raw=raw, z=z, zr=zr, zstd=zstd, fill=fill,
    )
    print(f"{MODALITY} modules {len(y)} anomalies {int(y.sum())} | fill<0.85: {int((fill < 0.85).sum())} "
          f"(anomalies among them {int(y[fill < 0.85].sum())})")
    truncated = fill < 0.85
    scores = {"raw": raw, "z": z, "zr(full)": np.where(truncated, 0.0, zr)}
    if RECTIFY:
        scores.update({"zstd": zstd, "zr+zstd": combined, "zr+zstd(full)": np.where(truncated, 0.0, combined)})
    print("ALL  " + " | ".join(f"{name} {auc(s, y):.3f}" for name, s in scores.items()))
    by_group = defaultdict(list)
    for i, group in enumerate(g):
        by_group[group].append(i)
    for group in sorted(by_group):
        idx = np.array(by_group[group])
        if y[idx].sum():
            print(f"{group} modules {len(idx)} anomalies {int(y[idx].sum())} "
                  + " | ".join(f"{name} {auc(s[idx], y[idx]):.3f}" for name, s in scores.items()))
    # Per-frame top-1: in frames with exactly one anomaly, is it the highest-scoring module?
    hits = total = 0
    for stem in np.unique(stems):
        idx = np.where(stems == stem)[0]
        if y[idx].sum() == 1:
            total += 1
            hits += int(y[idx][np.argmax(raw[idx])] == 1)
    print(f"one-anomaly frames: {total}, anomaly ranked first: {hits} ({hits / max(total, 1):.3f})")


if __name__ == "__main__":
    main()
