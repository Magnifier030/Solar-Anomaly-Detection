"""Recompute raw per-module mutual score and grey std on cached predicted modules, then test
robust z-scores with a scale floor (scale = max(1.4826*MAD, c*median)) so near-uniform frames
with few modules cannot blow up. Threshold and c chosen on calibration groups, applied to held-out.

Usage: python raw_eval.py ir|eo
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("module_maskrcnn", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

from rectify import rectify  # noqa: E402
from solar_ir.metrics.official import f1_tiou  # noqa: E402
from zs_score import FILL_MIN, SIZE, _mutual, _tokens, fill_ratio  # noqa: E402

MOD = sys.argv[1]
KTH = {"ir": 1, "eo": 2}[MOD]
ANOM = mm.ANOMALY[MOD]


def add_raw(items):
    folder = mm.TRAIN / mm.FOLDER[MOD] / "image"
    for n, item in enumerate(items, 1):
        mods = item["modules"]
        if len(mods) < 3:
            continue
        image = mm.load_image(mm.image_path(folder, item["name"]), MOD)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        polys = [np.asarray(m["polygon"], float) for m in mods]
        crops = [np.ascontiguousarray(rectify(image, p, SIZE, inset=0.03), np.uint8) for p in polys]
        raw = _mutual(_tokens(crops), KTH)
        for m, p, r in zip(mods, polys, raw):
            m["raw"] = float(r)
            m["std"] = float(rectify(grey, p, 64, inset=0.06).astype(np.float32).std())
            m["fill"] = float(fill_ratio(p))
        if n % 100 == 0:
            print(f"raw {n}/{len(items)}", flush=True)


def z(x, c):
    med = np.median(x)
    scale = max(np.median(np.abs(x - med)) * 1.4826, c * abs(med), 1e-6)
    return (x - med) / scale


UNION = len(sys.argv) > 3 and sys.argv[3] == "union"


def label(items, c, thr, use_std):
    samples = []
    for it in items:
        mods = [m for m in it["modules"] if m["det_score"] >= (0.5 if MOD == "ir" else 0.7)]
        pred = {}
        if len(mods) >= 3:
            raw = np.array([m["raw"] for m in mods])
            s = z(raw, c)
            if use_std:
                s = s + np.abs(z(np.array([m["std"] for m in mods]), c))
            s = np.where(np.array([m["fill"] for m in mods]) < FILL_MIN, 0, s)
        else:
            s = np.zeros(len(mods))
        for m, v in zip(mods, s):
            hit = v >= thr or (UNION and m["det_label"] == 2)
            pred.setdefault(ANOM if hit else "solar_panel", []).append(m["polygon"])
        samples.append({"ground_truth": it["truth"], "prediction": pred, "group": it["group"]})
    return samples


def score(samples):
    r = f1_tiou(samples, ("solar_panel", ANOM)).get("TIoU_" + ANOM, 0)
    per = {g: round(f1_tiou([s for s in samples if s["group"] == g], ("solar_panel", ANOM)).get("TIoU_" + ANOM, 0), 3)
           for g in sorted({s["group"] for s in samples})}
    return r, per


def main():
    sets = {}
    for tag, suffix in (("held", ""), ("cal", "_cal")):
        out = SCRATCH / f"raw_{MOD}{suffix}.json"
        if out.exists():
            sets[tag] = json.loads(out.read_text())
        else:
            sets[tag] = json.loads((SCRATCH / f"pipeline_{MOD}_module-maskrcnn-{MOD}-g013_k{KTH}{suffix}.json").read_text())
            add_raw(sets[tag])
            out.write_text(json.dumps(sets[tag]))
    use_std = MOD == "ir"
    grid = np.arange(0.5, 15.01, 0.25)
    for c in [float(x) for x in sys.argv[2].split(",")]:
        best = max(grid, key=lambda t: score(label(sets["cal"], c, t, use_std))[0])
        cal = score(label(sets["cal"], c, best, use_std))[0]
        held, per = score(label(sets["held"], c, best, use_std))
        oracle = max(score(label(sets["held"], c, t, use_std))[0] for t in grid[::2])
        print(f"c={c}: cal thr {best:.2f} cal {cal:.3f} -> held {held:.3f} {per} | oracle {oracle:.3f}", flush=True)


if __name__ == "__main__":
    main()
