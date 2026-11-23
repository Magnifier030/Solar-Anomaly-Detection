"""Audit IR submission geometry and compare validation/test model responses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from torch.utils.data import DataLoader

from research_training.backends import configure_backends
from research_training.storage import file_sha256
from solar_ir.inference import TestImages
from solar_ir.metrics import OfficialLocalizationMetric
from solar_ir.models.loader import load_model
from solar_ir.postprocess import mask_to_polygons
from solar_ir.task import SolarIRTask

THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]


def appearance(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    distance = float(
        np.abs(rgb[:, :, 0] - rgb[:, :, 1]).mean() + np.abs(rgb[:, :, 1] - rgb[:, :, 2]).mean()
    )
    return {
        "style": "gray" if distance < 1 else "color",
        "color_distance": distance,
        "visible_fraction": float((rgb.max(axis=2) > 5).mean()),
    }


def distribution(values):
    return (
        dict(
            zip(
                ["min", "p10", "p25", "p50", "p75", "p90", "max"],
                np.percentile(values, [0, 10, 25, 50, 75, 90, 100]).tolist(),
                strict=True,
            )
        )
        if values
        else {}
    )


def montage(rows, source_root, output, submission):
    canvas = Image.new("RGB", (1280, 840), "#202020")
    for i, row in enumerate(rows[:12]):
        key = row["key"]
        with Image.open(source_root / key) as image:
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in submission[key]:
            points = [tuple(p) for p in obj["points"]]
            draw.line(points + points[:1], fill=(255, 30, 30), width=2)
        x, y = (i % 4) * 320, (i // 4) * 280
        canvas.paste(image.resize((320, 256)), (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 3, y + 258), f"{Path(key).stem}: {len(submission[key])} predictions", fill="white"
        )
    canvas.save(output, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    root = Path(args.submission_dir)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    submission = json.loads((root / "submission.json").read_text())
    config = yaml.safe_load(Path(manifest["settings"]["training_config"]).read_text())
    configure_backends(config, device)
    task = SolarIRTask()
    bundle = task.build(config, device)
    best_path = Path(manifest["settings"]["checkpoint"])
    assert file_sha256(best_path) == manifest["checkpoint_sha256"]
    bundle.model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)["model"]
    )
    with Path(config["task"]["groups_csv"]).open() as stream:
        group_by_frame = {r["frame_id"]: r["group"] for r in csv.DictReader(stream)}
    test_root = Path(manifest["settings"]["test_root"])
    rows, areas = [], []
    for record in manifest["images"]:
        key = record["key"]
        polys = [Polygon(obj["points"]) for obj in submission[key]]
        area = [float(poly.area) for poly in polys]
        areas.extend(area)
        mask = np.asarray(Image.open(root / "ir_masks" / f"{Path(key).stem}.png")) > 0
        rgb = np.asarray(Image.open(test_root / key).convert("RGB"))
        rows.append(
            {
                **record,
                **appearance(test_root / key),
                "polygon_areas": area,
                "predicted_black_pixels": int((mask & (rgb.max(axis=2) <= 5)).sum()),
            }
        )
    summary = {
        "submission_sha256": file_sha256(root / "submission.json"),
        "style_definition": (
            "RGB mean absolute R-G plus G-B distance <1 => gray; descriptive heuristic"
        ),
        "polygon_areas_px": distribution(areas),
        "objects_area_lt10": sum(a < 10 for a in areas),
        "objects_area_lt100": sum(a < 100 for a in areas),
        "objects_area_lt1000": sum(a < 1000 for a in areas),
        "predicted_black_pixels": sum(r["predicted_black_pixels"] for r in rows),
        "test_by_style": {},
        "inference": {},
        "validation": {},
    }
    for style in ["gray", "color"]:
        records = [r for r in rows if r["style"] == style]
        summary["test_by_style"][style] = {
            "images": len(records),
            "images_with_predictions": sum(r["objects"] > 0 for r in records),
            "objects": sum(r["objects"] for r in records),
            "predicted_pixels": sum(r["foreground_pixels"] for r in records),
        }
        positive = sorted(
            [r for r in records if r["objects"]], key=lambda r: r["foreground_pixels"], reverse=True
        )
        negative = [r for r in records if not r["objects"]]
        montage(positive[:12], test_root, output / f"{style}_predictions.jpg", submission)
        selected = [negative[int(i * (len(negative) - 1) / 11)] for i in range(12)]
        montage(selected, test_root, output / f"{style}_no_predictions.jpg", submission)

    # Reference validation: group-wise pixel and official scores, including small G02 objects.
    bundle.model.eval()
    group_records = defaultdict(list)
    group_pixels = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0})
    validation_objects = defaultdict(list)
    offset = 0
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(bundle.validation_loader, 1):
            probs = bundle.model(images.to(device))[-1].sigmoid().cpu()
            for prob, target in zip(probs, targets, strict=True):
                path = bundle.validation_loader.dataset.image_paths[offset]
                group = group_by_frame[path.stem]
                pred, gt = prob >= 0.5, target > 0
                for key, value in {
                    "tp": int((pred & gt).sum()),
                    "fp": int((pred & ~gt).sum()),
                    "fn": int((~pred & gt).sum()),
                    "images": 1,
                }.items():
                    group_pixels[group][key] += value
                truths = mask_to_polygons(target[0].numpy())
                for polygon in truths:
                    validation_objects[group].append(Polygon(polygon).area)
                group_records[group].append(
                    {
                        "ground_truth": {"hotspot": truths},
                        "prediction": {"hotspot": mask_to_polygons(prob[0].numpy())},
                    }
                )
                offset += 1
            if batch_index % 20 == 0:
                print(f"validation {offset}/208", flush=True)
    for group, counts in group_pixels.items():
        union = counts["tp"] + counts["fp"] + counts["fn"]
        summary["validation"][group] = {
            **counts,
            "pixel_iou": counts["tp"] / union if union else 0,
            **OfficialLocalizationMetric(("hotspot",))(group_records[group]),
            "gt_polygon_areas_px": distribution(validation_objects[group]),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    test_loader = DataLoader(
        TestImages([r["key"] for r in rows], test_root, config["task"]), batch_size=4
    )
    pretrained_config = dict(config["task"]["model"])
    pretrained_config["pretrained_path"] = "outputs/pvhsd-dnanet-1ch-pretrain/best.pt"
    models = {
        "finetuned": bundle.model,
        "pretrained": load_model(pretrained_config).to(device).eval(),
    }
    for model_name, model in models.items():
        results = []
        offset = 0
        with torch.inference_mode():
            for batch_index, (images, keys, _width, _height) in enumerate(test_loader, 1):
                probs = model(images.to(device))[-1].sigmoid().cpu()
                for prob, key in zip(probs, keys, strict=True):
                    counts = {str(t): int((prob >= t).sum()) for t in THRESHOLDS}
                    row = {
                        "key": key,
                        "style": rows[offset]["style"],
                        "max_probability": float(prob.max()),
                        "mean_probability": float(prob.mean()),
                        "threshold_pixels": counts,
                    }
                    if model_name == "finetuned":
                        actual = (
                            np.asarray(Image.open(root / "ir_masks" / f"{Path(key).stem}.png")) > 0
                        )
                        row["replay_disagreement_pixels"] = int(
                            ((prob[0].numpy() >= 0.5) != actual).sum()
                        )
                    results.append(row)
                    offset += 1
                if batch_index == 1 or batch_index % 40 == 0:
                    print(f"{model_name} test {offset}/696", flush=True)
        summary["inference"][model_name] = {}
        for style in ["all", "gray", "color"]:
            selected = [r for r in results if style == "all" or r["style"] == style]
            summary["inference"][model_name][style] = {
                "max_probability_distribution": distribution(
                    [r["max_probability"] for r in selected]
                ),
                "thresholds": {
                    str(t): {
                        "images_with_foreground": sum(
                            r["threshold_pixels"][str(t)] > 0 for r in selected
                        ),
                        "foreground_pixels": sum(r["threshold_pixels"][str(t)] for r in selected),
                    }
                    for t in THRESHOLDS
                },
            }
        (output / f"{model_name}_test_probabilities.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "test_geometry.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
