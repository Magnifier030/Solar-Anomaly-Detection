"""EO (RGB) module-crop crack classifier: frozen DINOv2 features + linear probe, group split.

Every EO polygon (solar_panel or crack) is cropped, resized to 224 and embedded with
frozen DINOv2-with-registers-base (CLS + mean patch). A linear probe trained on the
training groups scores crack vs normal panel; AUC/recall/FPR on the validation groups.
Features are cached to the scratchpad so later probes do not re-embed.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Dinov2WithRegistersModel

torch.manual_seed(0)
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
ROOT = PROJECT / "data/raw/datasets/9_16_train_set/train/EO"
GROUPS_CSV = PROJECT / "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"
VAL = {"G01", "G03", "G07"}
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CACHE = SCRATCH / f"eo_module_feats_{SIZE}.npz"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

groups = {
    r["frame_id"]: r["group"]
    for r in csv.DictReader(open(GROUPS_CSV, encoding="utf-8", newline=""))
}


def frame_crops(label_path):
    image = Image.open(ROOT / "image" / (label_path.stem + ".jpg")).convert("RGB")
    annotation = json.loads(label_path.read_text(encoding="utf-8"))
    for shape in annotation["shapes"]:
        if shape["label"] not in {"solar_panel", "crack"} or len(shape["points"]) < 3:
            continue
        points = np.array(shape["points"], float)
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        w, h = x1 - x0, y1 - y0
        if w < 20 or h < 20:
            continue
        box = (
            max(0, x0 - 0.05 * w), max(0, y0 - 0.05 * h),
            min(image.width, x1 + 0.05 * w), min(image.height, y1 + 0.05 * h),
        )
        crop = image.crop(tuple(int(v) for v in box)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        yield np.asarray(crop, np.uint8), int(shape["label"] == "crack")


def embed():
    model = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base").cuda().eval()
    registers = model.config.num_register_tokens
    feats, labels, frame_groups, stems = [], [], [], []
    buffer = []

    def flush():
        batch = torch.from_numpy(np.stack([b[0] for b in buffer])).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            hidden = model(pixel_values=((batch - MEAN) / STD).cuda()).last_hidden_state
        patch = hidden[:, 1 + registers:]
        feats.append(torch.cat([hidden[:, 0], patch.mean(1)], dim=1).cpu().numpy().astype(np.float16))
        for _, label, group, stem in buffer:
            labels.append(label)
            frame_groups.append(group)
            stems.append(stem)
        buffer.clear()

    paths = sorted((ROOT / "label").glob("*.json"))
    for index, path in enumerate(paths, 1):
        for crop, label in frame_crops(path):
            buffer.append((crop, label, groups.get(path.stem, "?"), path.stem))
            if len(buffer) == 128:
                flush()
        if index % 200 == 0:
            print(f"embedded frames {index}/{len(paths)}", flush=True)
    if buffer:
        flush()
    np.savez(CACHE, X=np.concatenate(feats), y=np.array(labels), g=np.array(frame_groups), s=np.array(stems))


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    p = label.sum()
    n = len(label) - p
    return (ranks[label == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else float("nan")


def probe():
    data = np.load(CACHE)
    X, y, g = data["X"].astype(np.float32), data["y"], data["g"]
    is_val = np.isin(g, list(VAL))
    print("crops", len(y), "cracks", int(y.sum()), "| train cracks", int(y[~is_val].sum()),
          "val cracks", int(y[is_val].sum()), "val modules", int(is_val.sum()), flush=True)
    mu, sd = X[~is_val].mean(0), X[~is_val].std(0) + 1e-6
    Z = torch.tensor((X - mu) / sd)
    Y = torch.tensor(y, dtype=torch.float32)
    train = torch.tensor(~is_val)
    clf = torch.nn.Linear(Z.shape[1], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_w = torch.tensor((1 - y[~is_val]).sum() / max(y[~is_val].sum(), 1), dtype=torch.float32)
    for _ in range(1500):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            clf(Z[train]).squeeze(1), Y[train], pos_weight=pos_w
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.sigmoid(clf(Z)).squeeze(1).numpy()
    v = is_val
    pred = s[v] >= 0.5
    print("VAL AUC", round(auc(s[v], y[v]), 3), "| recall(crack)", round(float(pred[y[v] == 1].mean()), 3),
          "| FPR(panel)", round(float(pred[y[v] == 0].mean()), 3), flush=True)
    for group in sorted(VAL):
        m = v & (g == group)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(group, "n", int(m.sum()), "cracks", int(y[m].sum()), "AUC", round(auc(s[m], y[m]), 3))
    print("TRAIN AUC", round(auc(s[~v], y[~v]), 3), flush=True)
    np.save(SCRATCH / f"eo_module_scores_{SIZE}.npy", s)


if __name__ == "__main__":
    if not CACHE.exists():
        embed()
    probe()
