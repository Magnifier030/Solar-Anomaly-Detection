"""Cache per-module test predictions + raw zero-shot features so submissions can be rebuilt cheaply.

Per test image: modules from Mask R-CNN (class-agnostic select at floor 0.5) with polygon,
det_score, det_label, raw mutual score (kth), rectified grey std, fill ratio.
Usage: python test_cache.py ir|eo RUN_DIR
"""

from __future__ import annotations

import csv
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
from rectify import rectify  # noqa: E402
from zs_score import SIZE, _mutual, _tokens, fill_ratio  # noqa: E402

MOD, RUN = sys.argv[1], Path(sys.argv[2])
KTH = {"ir": 1, "eo": 2}[MOD]
TEST = PROJECT / "data/raw/datasets/9_30_test_set"
KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"


def main():
    with (KIT / "test_images.csv").open(encoding="utf-8-sig", newline="") as s:
        keys = [r["filename"] for r in csv.DictReader(s) if r["filename"].startswith(f"test/{mm.FOLDER[MOD]}/image/")]
    device = torch.device("cuda")
    model = mm.build_model(MOD, pretrained=False)
    model.load_state_dict(torch.load(RUN / "last.pt", map_location="cpu")["model"])
    model.to(device).eval()
    out = {}
    for i, key in enumerate(keys, 1):
        image = mm.load_image(TEST / key, MOD)
        kept = mm.select(mm.predict(model, image, device), 0.5, 0.5, 0.5)
        mods = [{"polygon": d["polygon"], "det_score": d["score"], "det_label": d["label"]} for d in kept]
        if len(mods) >= 3:
            grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            polys = [np.asarray(m["polygon"], float) for m in mods]
            crops = [np.ascontiguousarray(rectify(image, p, SIZE, inset=0.03), np.uint8) for p in polys]
            raw = _mutual(_tokens(crops), KTH)
            for m, p, r in zip(mods, polys, raw):
                m["raw"] = float(r)
                m["std"] = float(rectify(grey, p, 64, inset=0.06).astype(np.float32).std())
                m["fill"] = float(fill_ratio(p))
        out[key] = mods
        if i % 100 == 0:
            print(f"{MOD} {i}/{len(keys)}", flush=True)
    (SCRATCH / f"test_cache_{MOD}_{RUN.name}.json").write_text(json.dumps(out))
    print("done", flush=True)


if __name__ == "__main__":
    main()
