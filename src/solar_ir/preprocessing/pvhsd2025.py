"""Prepare one-channel PV-HSD images and explicit bbox-derived weak masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PIPELINE_VERSION = "grayscale-box-masks-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_mask(size: tuple[int, int], boxes: list[list[float]]) -> Image.Image:
    """Rasterize normalized boxes with half-open bounds; these are NOT true contours."""
    width, height = size
    array = np.zeros((height, width), dtype=np.uint8)
    for _, cx, cy, bw, bh in boxes:
        left = max(0, min(width, math.floor((cx - bw / 2) * width)))
        right = max(0, min(width, math.ceil((cx + bw / 2) * width)))
        top = max(0, min(height, math.floor((cy - bh / 2) * height)))
        bottom = max(0, min(height, math.ceil((cy + bh / 2) * height)))
        array[top:bottom, left:right] = 255
    return Image.fromarray(array)


def family_split(family: str, seed: int, fraction: float) -> str:
    """Keep the filename-family heuristic together, without using model performance."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{family}".encode()).digest()[:8], "big")
    return "validation" if value / 2**64 < fraction else "train"


def prepare(
    source_root: Path, destination: Path, seed: int = 7, validation_fraction: float = 0.2
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    receipt = json.loads((source_root / "download_receipt.json").read_text())
    candidates = json.loads((source_root / "ir_candidate_manifest.json").read_text())
    records = {r["image"]: r for r in json.loads((source_root / "audit_records.json").read_text())}
    # Preserve checksum linkage to the audited source labels, not just their parsed boxes.
    source_files = {
        r["path"]: r for r in json.loads((source_root / "file_manifest.json").read_text())
    }
    signature = {
        "pipeline": PIPELINE_VERSION,
        "source_revision": receipt["revision"],
        "source_audit_sha256": file_hash(source_root / "audit_records.json"),
        "source_file_manifest_sha256": file_hash(source_root / "file_manifest.json"),
        "source_candidate_manifest_sha256": file_hash(source_root / "ir_candidate_manifest.json"),
        "seed": seed,
        "validation_fraction": validation_fraction,
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Preparation changed: use a new versioned destination")
        print(f"Already prepared: {manifest_path}", flush=True)
        return previous
    if destination.exists():
        raise ValueError("Destination exists without a completed manifest; inspect before retrying")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise ValueError("Incomplete preparation exists; inspect partial output before retrying")
    (staging / "images").mkdir(parents=True)
    (staging / "masks").mkdir()
    prepared = []
    seen = set()
    for index, candidate in enumerate(candidates):
        record = records[candidate["image"]]
        source = source_root / "extracted" / record["image"]
        label = source_root / "extracted" / record["label"]
        if file_hash(source) != record["sha256"]:
            raise ValueError(f"Source image changed since audit: {source}")
        if file_hash(label) != source_files[record["label"]]["sha256"]:
            raise ValueError(f"Source annotation changed since audit: {label}")
        stem = source.stem
        if stem in seen:
            raise ValueError(f"Duplicate source stem: {stem}")
        seen.add(stem)
        with Image.open(source) as original:
            gray = original.convert("L")
        if gray.size != (640, 512):
            raise ValueError(f"Unexpected candidate image size: {source}")
        mask = box_mask(gray.size, record["boxes"])
        image_path, mask_path = Path("images") / f"{stem}.png", Path("masks") / f"{stem}.png"
        gray.save(staging / image_path)
        mask.save(staging / mask_path)
        family = candidate["filename_family_hint"]
        prepared.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "image_sha256": file_hash(staging / image_path),
                "mask_sha256": file_hash(staging / mask_path),
                "source_image": record["image"],
                "source_label": record["label"],
                "original_split": candidate["original_split"],
                "family_hint": family,
                "split": family_split(family, seed, validation_fraction),
                "boxes": len(record["boxes"]),
                "positive": bool(record["boxes"]),
            }
        )
        if (index + 1) % 500 == 0:
            print(f"Prepared {index + 1}/{len(candidates)} one-channel pairs", flush=True)
    splits = {}
    for split in ["train", "validation"]:
        rows = [r for r in prepared if r["split"] == split]
        if not rows or not any(r["positive"] for r in rows):
            raise ValueError(f"Empty or all-negative split: {split}")
        splits[split] = {
            "images": len(rows),
            "positive_images": sum(r["positive"] for r in rows),
            "boxes": sum(r["boxes"] for r in rows),
            "families": len({r["family_hint"] for r in rows}),
        }
    manifest = {
        "dataset": "PV-HSD-2025",
        "signature": signature,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_channels": 1,
        "image_encoding": "8-bit lossless PNG, PIL RGB-to-L luminance / 255",
        "temperature_recovered": False,
        "supervision": "bbox_raster_weak_mask",
        "mask_encoding": "L PNG, 0 background / 255 bbox union; no contour refinement",
        "split_policy": "sha256(seed:filename_family_hint), NOT verified physical scene groups",
        "source_archive_sha256": receipt["archive_sha256"],
        "summary": {
            "images": len(prepared),
            "boxes": sum(r["boxes"] for r in prepared),
            "original_split_counts": dict(Counter(r["original_split"] for r in prepared)),
            "splits": splits,
        },
        "samples": prepared,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.replace(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/external/PV-HSD-2025"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/pv_hsd2025/grayscale_box_masks_v1")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = prepare(args.source_root, args.output_root, args.seed, args.validation_fraction)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
