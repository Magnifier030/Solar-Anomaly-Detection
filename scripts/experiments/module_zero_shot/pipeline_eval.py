"""Full module pipeline on held-out groups: Mask R-CNN module instances + zero-shot anomaly labels.

Detections come from the other session's module Mask R-CNN (scripts/module_maskrcnn.py,
imported read-only). Kept modules (class-agnostic mask NMS) are then labelled:
  maskrcnn : the detector's own normal/anomaly class
  zeroshot : anomaly if the within-frame zero-shot score >= threshold (set on non-held-out groups)
  union    : anomaly if either says so
All variants are scored with the official F1_TIoU against the per-module GT polygons.

Usage: python pipeline_eval.py ir|eo RUN_DIR KTH THRESHOLD
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("module_maskrcnn", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

from solar_ir.metrics.official import f1_tiou  # noqa: E402
from zs_score import score_frame  # noqa: E402

MODALITY, RUN, KTH, THRESHOLD = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
# Optional: comma-separated calibration groups. Their frames are scored instead of the
# held-out ones, to pick the zero-shot threshold on predicted (not GT) polygons.
CALIBRATION = sys.argv[5].split(",") if len(sys.argv) > 5 else None
ANOMALY = mm.ANOMALY[MODALITY]
CACHE = SCRATCH / f"pipeline_{MODALITY}_{RUN.name}_k{KTH}{'_cal' if CALIBRATION else ''}.json"


def collect():
    device = torch.device("cuda")
    model = mm.build_model(MODALITY, pretrained=False)
    model.load_state_dict(torch.load(RUN / "last.pt", map_location="cpu")["model"])
    model.to(device).eval()
    config = json.loads((RUN / "config.json").read_text())
    _, names = mm.split(config["validation_groups"])
    groups = mm.frames_by_group()
    if CALIBRATION:
        # G02 has many frames with near-identical arrays; every third keeps the run short.
        names = [n for n in sorted(groups) if groups[n] in CALIBRATION]
        names = [n for i, n in enumerate(names) if groups[n] != "G02" or i % 3 == 0]
    items = []
    for index, name in enumerate(names, 1):
        image = mm.load_image(mm.image_path(mm.TRAIN / mm.FOLDER[MODALITY] / "image", name), MODALITY)
        detections = mm.predict(model, image, device)
        kept = mm.select(detections, 0.3, 0.3, 0.5)  # low floor; the module threshold is swept later
        polygons = [d["polygon"] for d in kept]
        zs = score_frame(image, polygons, KTH, use_std=MODALITY == "ir") if polygons else {"score": np.zeros(0)}
        truth = {}
        for label, points in mm.load_shapes(name, MODALITY):
            truth.setdefault(label, []).append(points.tolist())
        items.append({
            "name": name, "group": groups[name], "truth": truth,
            "modules": [
                {"polygon": d["polygon"], "det_score": d["score"], "det_label": d["label"], "zs": float(s)}
                for d, s in zip(kept, zs["score"])
            ],
        })
        if index % 50 == 0:
            print(f"frames {index}/{len(names)}", flush=True)
    CACHE.write_text(json.dumps(items))
    return items


def evaluate(items, module_threshold, mode, zs_threshold):
    samples = []
    for item in items:
        prediction = {}
        for m in item["modules"]:
            if m["det_score"] < module_threshold:
                continue
            by_det = m["det_label"] == 2
            by_zs = m["zs"] >= zs_threshold
            anomalous = {"maskrcnn": by_det, "zeroshot": by_zs, "union": by_det or by_zs}[mode]
            prediction.setdefault(ANOMALY if anomalous else "solar_panel", []).append(m["polygon"])
        samples.append({"ground_truth": item["truth"], "prediction": prediction, "group": item["group"]})
    result = {"all": f1_tiou(samples, ("solar_panel", ANOMALY))}
    for group in sorted({s["group"] for s in samples}):
        result[group] = f1_tiou([s for s in samples if s["group"] == group], ("solar_panel", ANOMALY))
    return result


def main():
    items = json.loads(CACHE.read_text()) if CACHE.exists() else collect()
    if CALIBRATION:
        best = max(
            np.arange(0.5, 12.01, 0.25),
            key=lambda t: evaluate(items, 0.5, "zeroshot", t)["all"].get("TIoU_" + ANOMALY, 0),
        )
        r = evaluate(items, 0.5, "zeroshot", best)["all"]
        print(f"calibration {CALIBRATION}: best zero-shot threshold {best:.2f} -> {ANOMALY} {r.get('TIoU_' + ANOMALY, 0):.3f}")
        return
    for module_threshold in (0.3, 0.5, 0.7):
        for mode in ("maskrcnn", "zeroshot", "union"):
            r = evaluate(items, module_threshold, mode, THRESHOLD)
            groups = " ".join(
                f"{g}:{r[g].get('TIoU_' + ANOMALY, 0):.3f}" for g in r if g != "all"
            )
            print(f"det>={module_threshold} {mode:9s} panel {r['all']['TIoU_solar_panel']:.3f} "
                  f"{ANOMALY} {r['all'].get('TIoU_' + ANOMALY, 0):.3f} F1(2cls) {r['all']['F1_TIoU']:.3f} "
                  f"| {ANOMALY} by group {groups}", flush=True)


if __name__ == "__main__":
    main()
