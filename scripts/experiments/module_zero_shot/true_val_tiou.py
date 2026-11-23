"""Official F1_TIoU of a DINO-seg IR checkpoint against the ORIGINAL per-module polygons.

The training-time validation rasterizes GT into masks and re-polygonizes them, which
merges adjacent modules into one object. Here GT comes straight from the labelme JSON
(one polygon per module), scaled to the 672 canvas like the predictions.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
sys.path.insert(0, str(ROOT / "src"))

from research_training.config import load_config  # noqa: E402
from solar_ir.fusion_task import create_task  # noqa: E402
from solar_ir.metrics.official import f1_tiou  # noqa: E402
from solar_ir.postprocess import mask_to_polygons  # noqa: E402

CONFIG = ROOT / sys.argv[1]
CHECKPOINT = ROOT / sys.argv[2]
LABELS = ROOT / "data/raw/datasets/9_16_train_set/train/IR/label"
CANVAS = 672
CLASSES = ("solar_panel", "hotspot")


def main():
    device = torch.device("cuda")
    task = create_task()
    bundle = task.build(load_config(CONFIG), device)
    bundle.model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    bundle.model.eval()
    frames = bundle.metadata["validation_frames"]
    samples_true, samples_merged = [], []
    offset = 0
    with torch.no_grad():
        for images, targets in bundle.validation_loader:
            probabilities = torch.sigmoid(bundle.model(images.to(device))).cpu().numpy()
            for probability, target in zip(probabilities, targets.numpy(), strict=True):
                stem = frames[offset]
                offset += 1
                annotation = json.loads((LABELS / f"{stem}.json").read_text(encoding="utf-8"))
                sx = CANVAS / annotation.get("imageWidth", 640)
                sy = CANVAS / annotation.get("imageHeight", 512)
                truth = {c: [] for c in CLASSES}
                for shape in annotation["shapes"]:
                    if shape["label"] in truth and len(shape["points"]) >= 3:
                        truth[shape["label"]].append([[x * sx, y * sy] for x, y in shape["points"]])
                prediction = {c: mask_to_polygons(probability[i], 0.5) for i, c in enumerate(CLASSES)}
                merged = {c: mask_to_polygons(target[i], 0.5) for i, c in enumerate(CLASSES)}
                samples_true.append({"ground_truth": truth, "prediction": prediction})
                samples_merged.append({"ground_truth": merged, "prediction": prediction})
    count = lambda s, c: sum(len(x["ground_truth"][c]) for x in s)  # noqa: E731
    print("GT objects  per-module:", {c: count(samples_true, c) for c in CLASSES},
          "| merged-mask:", {c: count(samples_merged, c) for c in CLASSES})
    print("pred objects:", {c: sum(len(x["prediction"][c]) for x in samples_true) for c in CLASSES})
    print("merged-mask GT (training-time):", {k: round(v, 4) for k, v in f1_tiou(samples_merged, CLASSES).items()})
    print("per-module GT (official):     ", {k: round(v, 4) for k, v in f1_tiou(samples_true, CLASSES).items()})


if __name__ == "__main__":
    main()
