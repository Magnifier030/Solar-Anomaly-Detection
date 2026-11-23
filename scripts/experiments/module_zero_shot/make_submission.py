"""Test submission: module Mask R-CNN instances + zero-shot within-frame anomaly labels.

For each test image (IR and EO scored separately), modules come from the modality's
Mask R-CNN (class-agnostic, mask NMS). Each module is `hotspot`/`crack` when its
within-frame zero-shot score reaches the threshold calibrated on non-held-out groups,
else `solar_panel`. Anomalies carry a fixed Traditional Chinese `reason`.

A threshold of `maskrcnn` instead of a number keeps the detector's own anomaly class
(used for EO, where it beat zero-shot on the held-out groups).

Usage: python make_submission.py OUTPUT_DIR IR_RUN IR_THR EO_RUN EO_THR
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("module_maskrcnn", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

from zs_score import score_frame  # noqa: E402

TEST = PROJECT / "data/raw/datasets/9_30_test_set"
KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"
MODULE_THRESHOLD = {"ir": 0.5, "eo": 0.7}
REASON = {
    # Chosen by text_opt.py (BERTScore proxy vs training reasons, platform settings).
    "hotspot": "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。",
    "crack": "模組板破裂，破損處的導電路徑受損。",
}
SETTINGS = {
    "ir": {"kth": 1, "use_std": True, "floor": float(os.environ.get("IR_FLOOR", 0))},
    "eo": {"kth": 2, "use_std": False, "floor": float(os.environ.get("EO_FLOOR", 0))},
}


def main():
    output = Path(sys.argv[1])
    runs = {"ir": (Path(sys.argv[2]), sys.argv[3]), "eo": (Path(sys.argv[4]), sys.argv[5])}
    output.mkdir(parents=True, exist_ok=False)
    with (KIT / "test_images.csv").open(encoding="utf-8-sig", newline="") as stream:
        keys = [row["filename"] for row in csv.DictReader(stream)]
    submission = {key: [] for key in keys}
    device = torch.device("cuda")
    summary = {}
    for modality, (run, threshold) in runs.items():
        model = mm.build_model(modality, pretrained=False)
        model.load_state_dict(torch.load(run / "last.pt", map_location="cpu")["model"])
        model.to(device).eval()
        prefix = f"test/{mm.FOLDER[modality]}/image/"
        anomaly = mm.ANOMALY[modality]
        counts = {"solar_panel": 0, anomaly: 0}
        modality_keys = [k for k in keys if k.startswith(prefix)]
        for index, key in enumerate(modality_keys, 1):
            image = mm.load_image(TEST / key, modality)  # IR pseudo-colour -> grey via inferno map
            floor = MODULE_THRESHOLD[modality]
            kept = mm.select(mm.predict(model, image, device), floor, floor, 0.5)
            polygons = [d["polygon"] for d in kept]
            if threshold == "maskrcnn":
                flags = [d["label"] == 2 for d in kept]
            else:
                scores = score_frame(image, polygons, **SETTINGS[modality])["score"] if polygons else []
                flags = [score >= float(threshold) for score in scores]
            objects = []
            for polygon, is_anomaly in zip(polygons, flags):
                if is_anomaly:
                    objects.append({"label": anomaly, "points": polygon, "reason": REASON[anomaly]})
                    counts[anomaly] += 1
                else:
                    objects.append({"label": "solar_panel", "points": polygon})
                    counts["solar_panel"] += 1
            submission[key] = objects
            if index % 100 == 0:
                print(f"{modality} {index}/{len(modality_keys)}", flush=True)
        summary[modality] = {"images": len(modality_keys), "run": str(run), "threshold": threshold, **counts}
        print(modality, summary[modality], flush=True)
        del model
        torch.cuda.empty_cache()
    path = output / "submission.json"
    path.write_text(json.dumps(submission, ensure_ascii=False, allow_nan=False) + "\n")
    checked = subprocess.run(
        [sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True
    )
    (output / "format_check.txt").write_text(checked.stdout + checked.stderr)
    (output / "manifest.json").write_text(json.dumps({
        "summary": summary, "module_threshold": MODULE_THRESHOLD, "reason": REASON, "settings": SETTINGS,
        "format_check_exit_code": checked.returncode, "uploaded": False,
    }, ensure_ascii=False, indent=1))
    print(checked.stdout, checked.stderr, flush=True)


if __name__ == "__main__":
    main()
