"""EO crack recall: keep class-agnostic modules, relabel a module as anomaly when any class-2
detection overlapping it (polygon IoU >= 0.5) has score >= t. Evaluated on held-out groups.
Usage: python relabel_eval.py ir|eo RUN_DIR
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("module_maskrcnn", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
from solar_ir.metrics.official import f1_tiou  # noqa: E402

MOD, RUN = sys.argv[1], Path(sys.argv[2])
ANOM = mm.ANOMALY[MOD]
CACHE = SCRATCH / f"relabel_{MOD}_{RUN.name}.json"


def mask(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.asarray(poly) / 4)], 1)
    return m.astype(bool)


def collect():
    device = torch.device("cuda")
    model = mm.build_model(MOD, pretrained=False)
    model.load_state_dict(torch.load(RUN / "last.pt", map_location="cpu")["model"])
    model.to(device).eval()
    config = json.loads((RUN / "config.json").read_text())
    _, names = mm.split(config["validation_groups"])
    groups = mm.frames_by_group()
    items = []
    for i, name in enumerate(names, 1):
        image = mm.load_image(mm.image_path(mm.TRAIN / mm.FOLDER[MOD] / "image", name), MOD)
        dets = mm.predict(model, image, device)
        kept = mm.select(dets, 0.5, 0.5, 0.5)
        shape = (image.shape[0] // 4 + 1, image.shape[1] // 4 + 1)
        anomalies = [(d["score"], mask(d["polygon"], shape)) for d in dets if d["label"] == 2 and d["score"] >= 0.02]
        modules = []
        for d in kept:
            km = mask(d["polygon"], shape)
            best = 0.0
            for s, am in anomalies:
                inter = (km & am).sum()
                if inter and inter / (km | am).sum() >= 0.5:
                    best = max(best, s)
            modules.append({"polygon": d["polygon"], "det_score": d["score"], "det_label": d["label"], "anom": best})
        truth = {}
        for label, pts in mm.load_shapes(name, MOD):
            truth.setdefault(label, []).append(pts.tolist())
        items.append({"name": name, "group": groups[name], "truth": truth, "modules": modules})
        if i % 50 == 0:
            print(f"frames {i}/{len(names)}", flush=True)
    CACHE.write_text(json.dumps(items))
    return items


def evaluate(items, floor, t):
    samples = []
    for it in items:
        pred = {}
        for m in it["modules"]:
            if m["det_score"] < floor:
                continue
            anomalous = m["det_label"] == 2 or m["anom"] >= t
            pred.setdefault(ANOM if anomalous else "solar_panel", []).append(m["polygon"])
        samples.append({"ground_truth": it["truth"], "prediction": pred, "group": it["group"]})
    r = f1_tiou(samples, ("solar_panel", ANOM))
    per = {g: round(f1_tiou([s for s in samples if s["group"] == g], ("solar_panel", ANOM)).get("TIoU_" + ANOM, 0), 3)
           for g in sorted({s["group"] for s in samples})}
    n = sum(len(p) for s in samples for k, p in s["prediction"].items() if k == ANOM)
    return r, per, n


def main():
    items = json.loads(CACHE.read_text()) if CACHE.exists() else collect()
    for floor in (0.7, 0.8):
        for t in (1.01, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
            r, per, n = evaluate(items, floor, t)
            print(f"det>={floor} relabel t={t}: panel {r['TIoU_solar_panel']:.3f} {ANOM} {r.get('TIoU_' + ANOM, 0):.3f} "
                  f"n_pred {n} {per}", flush=True)


if __name__ == "__main__":
    main()
