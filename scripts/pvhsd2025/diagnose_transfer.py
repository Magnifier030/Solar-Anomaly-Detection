"""Evaluate transferred checkpoints with pixel and official metric components."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from research_training.storage import file_sha256
from solar_ir.metrics.official import _class_components, _polygon
from solar_ir.models.dnanet import create_model
from solar_ir.postprocess import mask_to_polygons
from solar_ir.task import SolarIRTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reuse-snapshots", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run = Path("outputs") / config["name"]
    # Snapshot atomically published files, so ongoing training cannot change inputs.
    sources = {
        "pretrained": Path(config["task"]["model"]["pretrained_path"]),
        "finetuned": run / "latest.pt",
    }
    snapshots = {}
    for name, source in sources.items():
        destination = output / f"{name}_snapshot.pt"
        if args.reuse_snapshots:
            if not destination.is_file():
                raise FileNotFoundError(destination)
            snapshots[name] = destination
            continue
        with source.open("rb") as reader, destination.open("wb") as writer:
            import shutil

            shutil.copyfileobj(reader, writer)
        snapshots[name] = destination
    build_config = copy.deepcopy(config)
    build_config["task"]["model"].pop("pretrained_path", None)
    bundle = SolarIRTask().build(build_config, torch.device("cpu"))
    with Path(config["task"]["groups_csv"]).open() as stream:
        groups = {row["frame_id"]: row["group"] for row in csv.DictReader(stream)}
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": args.config,
        "samples": len(bundle.validation_loader.dataset),
        "device": str(device),
        "models": {},
    }
    for name, checkpoint in snapshots.items():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = create_model(build_config["task"]["model"]).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
        totals = {str(t): dict(tp=0, fp=0, fn=0, predicted_pixels=0) for t in thresholds}
        recall_sum = precision_sum = 0.0
        truth_count = prediction_count = offset = 0
        rows = []
        for batch_index, (images, masks) in enumerate(bundle.validation_loader, 1):
            with torch.inference_mode():
                probability = model(images.to(device))[-1].sigmoid().cpu()
            for prob, mask in zip(probability, masks, strict=True):
                frame = bundle.validation_loader.dataset.image_paths[offset].stem
                row = {
                    "frame": frame,
                    "group": groups[frame],
                    "target_pixels": int((mask > 0).sum()),
                    "probability_max": float(prob.max()),
                    "target_probability_sum": float(prob[mask > 0].sum()),
                }
                target = mask > 0
                for threshold in thresholds:
                    pred = prob >= threshold
                    counts = {
                        "tp": int((pred & target).sum()),
                        "fp": int((pred & ~target).sum()),
                        "fn": int((~pred & target).sum()),
                        "predicted_pixels": int(pred.sum()),
                    }
                    for key, value in counts.items():
                        totals[str(threshold)][key] += value
                    if threshold == 0.5:
                        row.update(counts)
                truths = [_polygon(p) for p in mask_to_polygons(mask[0].numpy(), 0.5)]
                predictions = [_polygon(p) for p in mask_to_polygons(prob[0].numpy(), 0.5)]
                recall, precision, ngt, npred = _class_components(truths, predictions)
                recall_sum += recall
                precision_sum += precision
                truth_count += ngt
                prediction_count += npred
                row.update(gt_polygons=ngt, predicted_polygons=npred)
                rows.append(row)
                offset += 1
            if batch_index == 1 or batch_index % 10 == 0:
                print(f"{name}: {offset}/{len(bundle.validation_loader.dataset)}", flush=True)
        for counts in totals.values():
            counts["intersection"] = counts["tp"]
            counts["union"] = counts["tp"] + counts["fp"] + counts["fn"]
            counts["iou"] = counts["tp"] / counts["union"] if counts["union"] else 0.0
            counts["dice"] = (
                2 * counts["tp"] / max(1, 2 * counts["tp"] + counts["fp"] + counts["fn"])
            )
        recall = recall_sum / truth_count if truth_count else 0.0
        precision = precision_sum / prediction_count if prediction_count else 0.0
        result["models"][name] = {
            "source": str(sources[name]),
            "snapshot_sha256": file_sha256(checkpoint),
            "checkpoint_epoch_raw": state["epoch"],
            "step": state.get("global_step"),
            "thresholds": totals,
            "official": {
                "recall_numerator": recall_sum,
                "recall_denominator": truth_count,
                "precision_numerator": precision_sum,
                "precision_denominator": prediction_count,
                "recall": recall,
                "precision": precision,
                "F1_TIoU": 2 * recall * precision / (recall + precision)
                if recall + precision
                else 0.0,
            },
            "images": rows,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, json.dumps(result["models"][name]["official"]), flush=True)


if __name__ == "__main__":
    main()
