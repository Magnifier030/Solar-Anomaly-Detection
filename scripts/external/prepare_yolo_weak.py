"""Convert a YOLO-box thermal dataset into the one-channel weak-mask layout used by PV-HSD pretraining.

Images are luminance-converted (PIL RGB->L), resized to 640x512, and boxes are rasterised with the
same half-open rule as PV-HSD. Splits are assigned by a SHA-256 hash of the original-source family,
so augmented copies of one photo never straddle train and validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from solar_ir.preprocessing.pvhsd2025 import box_mask, file_hash

SIZE = (640, 512)


def family_of(stem: str) -> str:
    return re.split(r"\.rf\.", stem)[0]


def family_split(family: str, seed: int, fraction: float) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{family}".encode()).digest()[:8], "big")
    return "validation" if value / 2**64 < fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="dir containing {train,valid,test}/{images,labels}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    out = args.output
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    samples = []
    for split_dir in ["train", "valid", "test"]:
        for img_path in sorted((args.source / split_dir / "images").glob("*.jpg")):
            stem = img_path.stem
            label = args.source / split_dir / "labels" / f"{stem}.txt"
            boxes = []
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        boxes.append([float(x) for x in parts])
            with Image.open(img_path) as im:
                gray = im.convert("L").resize(SIZE, Image.Resampling.BILINEAR)
            mask = box_mask(SIZE, boxes)
            image_rel, mask_rel = Path("images") / f"{stem}.png", Path("masks") / f"{stem}.png"
            gray.save(out / image_rel)
            mask.save(out / mask_rel)
            family = family_of(stem)
            samples.append({
                "image": str(image_rel), "mask": str(mask_rel),
                "image_sha256": file_hash(out / image_rel), "mask_sha256": file_hash(out / mask_rel),
                "source_image": f"{split_dir}/images/{img_path.name}", "original_split": split_dir,
                "family_hint": family, "split": family_split(family, args.seed, args.validation_fraction),
                "boxes": len(boxes), "positive": bool(boxes),
            })
    summary = {}
    for s in ["train", "validation"]:
        rows = [r for r in samples if r["split"] == s]
        summary[s] = {"images": len(rows), "positive_images": sum(r["positive"] for r in rows),
                      "boxes": sum(r["boxes"] for r in rows), "families": len({r["family_hint"] for r in rows})}
    manifest = {
        "dataset": args.name,
        "signature": {"pipeline": "yolo-box-weak-mask-v1", "source_revision": args.source_sha256[:12],
                      "seed": args.seed, "validation_fraction": args.validation_fraction},
        "input_channels": 1,
        "image_encoding": "8-bit lossless PNG, PIL RGB-to-L luminance, resized to 640x512",
        "temperature_recovered": False,
        "supervision": "bbox_raster_weak_mask",
        "split_policy": "sha256(seed:source_family), family = filename before '.rf.' (Roboflow augmentation id)",
        "source_archive_sha256": args.source_sha256,
        "summary": summary,
        "samples": samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
