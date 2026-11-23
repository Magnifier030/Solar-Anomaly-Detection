"""Audit paired solar groups and export reproducible statistics and contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def quantiles(values):
    return (
        dict(
            zip(
                ["min", "p10", "p50", "p90", "max"],
                np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                strict=True,
            )
        )
        if values
        else {}
    )


def analyze_annotation(path):
    a = json.loads(path.read_text())
    w, h = a["imageWidth"], a["imageHeight"]
    objects = []
    for s in a["shapes"]:
        pts = s["points"]
        if len(pts) < 3:
            continue
        area = (
            abs(
                sum(
                    x * pts[(i + 1) % len(pts)][1] - y * pts[(i + 1) % len(pts)][0]
                    for i, (x, y) in enumerate(pts)
                )
            )
            / 2
        )
        xs, ys = zip(*pts, strict=True)
        objects.append(
            {
                "label": s["label"],
                "area_fraction": area / (w * h),
                "bbox_256": [(max(xs) - min(xs)) * 256 / w, (max(ys) - min(ys)) * 256 / h],
                "reason": s.get("reason"),
                "points": pts,
            }
        )
    return {"size": [w, h], "objects": objects}


def summarize(records, anomaly):
    positives = [r for r in records if any(o["label"] == anomaly for o in r["objects"])]
    abnormal = [o for r in records for o in r["objects"] if o["label"] == anomaly]
    normal = [o for r in records for o in r["objects"] if o["label"] == "solar_panel"]
    return {
        "frames": len(records),
        "positive_frames": len(positives),
        "positive_fraction": len(positives) / len(records),
        "resolution_counts": dict(Counter(f"{r['size'][0]}x{r['size'][1]}" for r in records)),
        "labels": dict(Counter(o["label"] for r in records for o in r["objects"])),
        "anomaly_objects_per_positive_frame": quantiles(
            [sum(o["label"] == anomaly for o in r["objects"]) for r in positives]
        ),
        "anomaly_area_fraction": quantiles([o["area_fraction"] for o in abnormal]),
        "normal_panel_area_fraction": quantiles([o["area_fraction"] for o in normal]),
        "anomaly_bbox_width_at_256": quantiles([o["bbox_256"][0] for o in abnormal]),
        "anomaly_bbox_height_at_256": quantiles([o["bbox_256"][1] for o in abnormal]),
        "anomaly_bbox_short_edge_at_256": quantiles([min(o["bbox_256"]) for o in abnormal]),
        "anomaly_short_edge_below_4px_at_256": sum(min(o["bbox_256"]) < 4 for o in abnormal),
        "anomaly_short_edge_below_8px_at_256": sum(min(o["bbox_256"]) < 8 for o in abnormal),
        "reason_counts": dict(Counter(o["reason"] or "(missing)" for o in abnormal)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/raw/datasets/9_16_train_set")
    )
    parser.add_argument(
        "--groups-csv",
        type=Path,
        default=Path("data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/group_analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.groups_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected_frames = {r["frame_id"] for r in rows}
    if len(expected_frames) != len(rows):
        raise ValueError("Duplicate frame IDs in group CSV")
    for modality in ["IR", "EO"]:
        base = args.dataset_root / "train" / modality
        for folder, suffix in [("image", "jpg"), ("label", "json")]:
            actual = {p.stem for p in (base / folder).glob(f"*.{suffix}")}
            if actual != expected_frames:
                raise ValueError(f"Group CSV coverage mismatch: {modality}/{folder}")
    records = defaultdict(lambda: defaultdict(list))
    hashes = defaultdict(list)
    all_records = {}
    for row in rows:
        frame, group = row["frame_id"], row["group"]
        for modality in ["IR", "EO"]:
            base = args.dataset_root / "train" / modality
            r = analyze_annotation(base / "label" / f"{frame}.json")
            r.update(frame=frame, group=group, image_path=str(base / "image" / f"{frame}.jpg"))
            with Image.open(r["image_path"]) as image:
                if list(image.size) != r["size"]:
                    raise ValueError(f"Image/annotation size mismatch: {frame} {modality}")
            hashes[
                (modality, hashlib.sha256(Path(r["image_path"]).read_bytes()).hexdigest())
            ].append((group, frame))
            records[group][modality].append(r)
            all_records[(modality, frame)] = r
    report = {
        "definitions": {
            "area": "shoelace polygon area / original full image area; per object",
            "bbox_256": "bounding box after full-image resize to 256x256; no augmentation",
            "positive": "frame with >=1 hotspot (IR) or crack (EO)",
            "independence": "frame and object counts are not independent physical panels",
        },
        "groups": {},
        "totals": {},
    }
    table = []
    for group in sorted(records):
        entry = {
            m: summarize(records[group][m], "hotspot" if m == "IR" else "crack")
            for m in ["IR", "EO"]
        }
        pair_counts = Counter()
        for ir in records[group]["IR"]:
            eo = all_records[("EO", ir["frame"])]
            ip = any(o["label"] == "hotspot" for o in ir["objects"])
            ep = any(o["label"] == "crack" for o in eo["objects"])
            pair_counts[f"IR_{int(ip)}_EO_{int(ep)}"] += 1
        entry["paired_status_counts"] = dict(pair_counts)
        report["groups"][group] = entry
        table.append(
            {
                "group": group,
                "frames": entry["IR"]["frames"],
                **{
                    f"{m}_{k}": entry[m][k]
                    for m in ["IR", "EO"]
                    for k in ["positive_frames", "positive_fraction"]
                },
                **{
                    f"{m}_objects": entry[m]["labels"].get("hotspot" if m == "IR" else "crack", 0)
                    for m in ["IR", "EO"]
                },
                **{
                    f"{m}_area_median_pct": 100 * entry[m]["anomaly_area_fraction"].get("p50", 0)
                    for m in ["IR", "EO"]
                },
            }
        )
    for m, label in [("IR", "hotspot"), ("EO", "crack")]:
        report["totals"][m] = summarize([r for g in records.values() for r in g[m]], label)
    report["integrity"] = {
        "group_rows": len(rows),
        "unique_frames": len({r["frame_id"] for r in rows}),
        "exact_duplicate_image_sets": [
            {"modality": m, "frames": frames}
            for (m, _), frames in hashes.items()
            if len(frames) > 1
        ],
    }
    (args.output_dir / "statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    # Three representative IR frames (low/median/high target scale), EO paired to middle.
    sheet = Image.new("RGB", (1280, 220 * len(records)), "#15191f")
    draw = ImageDraw.Draw(sheet)
    selections = {}
    for row, group in enumerate(sorted(records)):
        candidates = sorted(records[group]["IR"], key=lambda r: r["frame"])
        positives = [r for r in candidates if any(o["label"] == "hotspot" for o in r["objects"])]
        candidates = (
            sorted(
                positives,
                key=lambda r: np.median(
                    [o["area_fraction"] for o in r["objects"] if o["label"] == "hotspot"]
                ),
            )
            if positives
            else candidates
        )
        if not positives:
            eo_positives = [
                r for r in records[group]["EO"] if any(o["label"] == "crack" for o in r["objects"])
            ]
            if eo_positives:
                candidates = [
                    all_records[("IR", r["frame"])]
                    for r in sorted(eo_positives, key=lambda r: r["frame"])
                ]
        chosen = [candidates[i] for i in [0, len(candidates) // 2, len(candidates) - 1]]
        chosen.append(all_records[("EO", chosen[1]["frame"])])
        selections[group] = [r["frame"] for r in chosen]
        for col, r in enumerate(chosen):
            image = Image.open(r["image_path"]).convert("RGB")
            image.thumbnail((316, 188))
            x, y = col * 320, row * 220
            sheet.paste(image, (x, y + 25))
            overlay = ImageDraw.Draw(sheet)
            sx, sy = image.width / r["size"][0], image.height / r["size"][1]
            for obj in r["objects"]:
                if obj["label"] == "solar_panel":
                    continue
                pts = [(x + px * sx, y + 25 + py * sy) for px, py in obj["points"]]
                overlay.line(pts + [pts[0]], fill="#ff4444", width=2)
            draw.text(
                (x + 3, y + 5), f"{group} {'IR' if col < 3 else 'EO'} {r['frame']}", fill="white"
            )
    sheet.save(args.output_dir / "group_contact_sheet.jpg", quality=92)
    (args.output_dir / "representatives.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(
        json.dumps(
            {"totals": report["totals"], "summary": table, "integrity": report["integrity"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
