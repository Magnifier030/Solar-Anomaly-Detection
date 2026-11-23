"""Report original polygon sizes by group and by the configured split (no augmentation)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from solar_ir.datasets import build_group_split


def summarize(records):
    areas = [a for r in records for a in r["areas"]]
    return {
        "frames": len(records),
        "positive_frames": sum(bool(r["areas"]) for r in records),
        "objects": len(areas),
        "polygon_area_fraction_quantiles": (
            dict(
                zip(
                    ["p10", "p50", "p90"], np.quantile(areas, [0.1, 0.5, 0.9]).tolist(), strict=True
                )
            )
            if areas
            else {}
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/solar_ir_dnanet.yaml")
    parser.add_argument("--output", default="outputs/split_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    task = config["task"]
    root = Path(task["dataset_root"]) / "train/IR"
    with Path(task["groups_csv"]).open(newline="") as stream:
        groups = {r["frame_id"]: r["group"] for r in csv.DictReader(stream)}
    train, validation = build_group_split(
        root / "image",
        root / "label",
        Path(task["groups_csv"]),
        task["validation_fraction"],
        task["split_strategy"],
        config["training"]["seed"],
        validation_groups=task.get("validation_groups"),
    )
    records = []
    for path in sorted((root / "label").glob("*.json")):
        ann = json.loads(path.read_text())
        areas = []
        for shape in ann["shapes"]:
            if shape["label"] != "hotspot":
                continue
            points = shape["points"]
            area = (
                abs(
                    sum(
                        x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
                        for i, (x, y) in enumerate(points)
                    )
                )
                / 2
            )
            areas.append(area / (ann["imageWidth"] * ann["imageHeight"]))
        records.append({"frame": path.stem, "group": groups[path.stem], "areas": areas})
    report = {
        "config": args.config,
        "area_definition": "polygon shoelace area / original image area",
        "groups": {
            g: summarize([r for r in records if r["group"] == g])
            for g in sorted(set(groups.values()))
        },
    }
    for name, paths in [("train", train), ("validation", validation)]:
        frames = {p.stem for p in paths}
        selected = [r for r in records if r["frame"] in frames]
        report[name] = {**summarize(selected), "groups": sorted({r["group"] for r in selected})}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
