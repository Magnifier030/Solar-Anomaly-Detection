"""Audit a downloaded PV-HSD-2025 YOLO dataset without changing source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def split_name(path: Path) -> str:
    return next((p for p in path.parts if p in {"train", "val", "valid", "test"}), "unspecified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/external/PV-HSD-2025"))
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("docs/datasets/PV-HSD-2025-exclusions.json"),
    )
    args = parser.parse_args()
    root = args.root
    extracted = root / "extracted"
    if not (root / "download_receipt.json").exists():
        raise ValueError("Run scripts/pvhsd2025/download.py successfully first")
    paths = sorted(p for p in extracted.rglob("*") if p.is_file())
    image_paths = [
        p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]
    labels = [p for p in paths if p.suffix.lower() == ".txt" and "labels" in p.parts]
    if not image_paths or not labels:
        raise ValueError("Expected extracted images and labels; inspect archive layout")
    label_by_stem = defaultdict(list)
    for path in labels:
        label_by_stem[path.stem].append(path)
    records = []
    hashes = defaultdict(list)
    pixels = defaultdict(list)
    used_labels = set()
    issues = []
    class_counts = Counter()
    counts = defaultdict(Counter)
    manifest = []
    for index, path in enumerate(image_paths):
        relative = path.relative_to(extracted)
        parts = list(relative.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label = (extracted / Path(*parts)).with_suffix(".txt")
        else:
            matches = label_by_stem[path.stem]
            label = matches[0] if len(matches) == 1 else None
        if label is None or not label.exists():
            issues.append({"image": str(relative), "error": "missing or ambiguous label"})
            continue
        used_labels.add(label)
        boxes = []
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                values = [float(v) for v in line.split()]
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError("Expected five finite YOLO detection values")
                cls, cx, cy, bw, bh = values
                if cls < 0 or int(cls) != cls or not (0 <= cx <= 1 and 0 <= cy <= 1):
                    raise ValueError("Invalid class or normalized center")
                if not (0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError("Invalid normalized box dimensions")
                boxes.append(values)
                class_counts[int(cls)] += 1
            except ValueError as error:
                issues.append(
                    {
                        "label": str(label.relative_to(extracted)),
                        "line": line_number,
                        "error": str(error),
                    }
                )
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with Image.open(path) as image:
            image.load()  # Fully decode every image, not just its header.
            size, mode = image.size, image.mode
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        hashes[digest].append(str(relative))
        pixels[pixel_hash].append(str(relative))
        split = split_name(relative)
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(boxes)
        counts[split]["positive_images"] += bool(boxes)
        counts[split]["empty_labels"] += not boxes
        record = {
            "image": str(relative),
            "label": str(label.relative_to(extracted)),
            "split": split,
            "size": list(size),
            "mode": mode,
            "boxes": boxes,
            "sha256": digest,
            "decoded_rgb_sha256": pixel_hash,
        }
        records.append(record)
        manifest.extend(
            [
                {"path": str(relative), "bytes": len(raw), "sha256": digest},
                {
                    "path": record["label"],
                    "bytes": label.stat().st_size,
                    "sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                },
            ]
        )
        if (index + 1) % 500 == 0:
            print(f"Decoded and audited {index + 1}/{len(image_paths)} images", flush=True)
    orphan_labels = [str(p.relative_to(extracted)) for p in labels if p not in used_labels]
    if orphan_labels:
        issues.append({"error": "orphan labels", "paths": orphan_labels})
    all_boxes = [box for r in records for box in r["boxes"]]
    summary = {
        "dataset": "PV-HSD-2025",
        "source_revision": json.loads((root / "download_receipt.json").read_text())["revision"],
        "image_count": len(records),
        "label_files": len(labels),
        "box_count": len(all_boxes),
        "split_counts": {k: dict(v) for k, v in counts.items()},
        "class_counts": dict(class_counts),
        "image_resolutions": dict(Counter(f"{r['size'][0]}x{r['size'][1]}" for r in records)),
        "image_modes": dict(Counter(r["mode"] for r in records)),
        "file_extensions": dict(Counter(p.suffix.lower() for p in paths)),
        "box_area_fraction_quantiles": dict(
            zip(
                ["p10", "p50", "p90"],
                np.quantile([b[3] * b[4] for b in all_boxes], [0.1, 0.5, 0.9]).tolist(),
                strict=True,
            )
        ),
        "boxes_crossing_image_boundary": sum(
            min(b[1] - b[3] / 2, b[2] - b[4] / 2) < -1e-6
            or max(b[1] + b[3] / 2, b[2] + b[4] / 2) > 1 + 1e-6
            for b in all_boxes
        ),
        "exact_file_duplicate_sets": [v for v in hashes.values() if len(v) > 1],
        "decoded_rgb_duplicate_sets": [v for v in pixels.values() if len(v) > 1],
        "issues": issues,
        "usage_status": "downloaded_and_audited_not_used_in_training",
        "mask_status": "no_masks_generated; original YOLO bbox supervision only",
    }
    exclusions = json.loads(args.exclusions.read_text()) if args.exclusions.exists() else {}
    if exclusions and exclusions["source_revision"] != summary["source_revision"]:
        raise ValueError("Manual exclusions were reviewed against a different dataset revision")
    excluded = {entry["image"] for entry in exclusions.get("exclusions", [])}
    if excluded - {record["image"] for record in records}:
        raise ValueError("Manual exclusion references an unknown image")
    summary["manual_exclusions"] = exclusions.get("exclusions", [])
    summary["ir_candidate_images"] = len(records) - len(excluded)
    candidates_manifest = []
    families = defaultdict(list)
    for record in records:
        family = Path(record["image"]).stem.rsplit("_", 1)[0]
        families[family].append(record)
        if record["image"] not in excluded:
            candidates_manifest.append(
                {
                    "image": record["image"],
                    "label": record["label"],
                    "original_split": record["split"],
                    "filename_family_hint": family,
                }
            )
    summary["filename_family_count"] = len(families)
    summary["filename_families_crossing_splits"] = sum(
        len({record["split"] for record in family}) > 1 for family in families.values()
    )
    summary["filename_family_caveat"] = (
        "Prefix before final underscore is an unverified filename heuristic, not an official "
        "scene/group identifier; near-duplicate audit is still required."
    )
    (root / "ir_candidate_manifest.json").write_text(
        json.dumps(candidates_manifest, indent=2) + "\n"
    )
    for kind in ["exact_file_duplicate_sets", "decoded_rgb_duplicate_sets"]:
        summary[kind.replace("sets", "cross_split_sets")] = [
            v for v in summary[kind] if len({split_name(Path(p)) for p in v}) > 1
        ]
    manifested_paths = {record["path"] for record in manifest}
    for path in paths:
        if str(path.relative_to(extracted)) not in manifested_paths:
            manifest.append(
                {
                    "path": str(path.relative_to(extracted)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (root / "file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "audit_records.json").write_text(json.dumps(records, indent=2) + "\n")
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Samples from low/median/high object area, selected deterministically.
    candidates = sorted(
        (r for r in records if r["boxes"]),
        key=lambda r: (np.median([b[3] * b[4] for b in r["boxes"]]), r["image"]),
    )
    chosen = [candidates[round((len(candidates) - 1) * q)] for q in np.linspace(0, 1, 9)]
    sheet = Image.new("RGB", (960, 3 * 292), "#171b21")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(chosen):
        with Image.open(extracted / record["image"]) as source:
            image = ImageOps.contain(source.convert("RGB"), (316, 253))
        x, y = (i % 3) * 320, (i // 3) * 292
        sheet.paste(image, (x, y + 32))
        draw.text((x + 3, y + 3), f"{record['split']}  boxes={len(record['boxes'])}", fill="white")
        for _, cx, cy, bw, bh in record["boxes"]:
            draw.rectangle(
                (
                    x + (cx - bw / 2) * image.width,
                    y + 32 + (cy - bh / 2) * image.height,
                    x + (cx + bw / 2) * image.width,
                    y + 32 + (cy + bh / 2) * image.height,
                ),
                outline="#ff4444",
                width=2,
            )
    sheet.save(root / "bbox_samples.jpg", quality=92)
    (root / "bbox_samples.json").write_text(
        json.dumps([r["image"] for r in chosen], indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if issues:
        raise ValueError("Audit found issues; see audit_summary.json")


if __name__ == "__main__":
    main()
