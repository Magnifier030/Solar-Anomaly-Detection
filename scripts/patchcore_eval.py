"""PatchCore one-class baseline for IR hotspot localization.

The protocol matches the supervised DNA-Net runs: the same leave-one-hotspot-group-out
folds (configs/external/finetune_fold_manifest.json) or the same photo-stratified split,
native 640x512 resolution, and the official F1_TIoU polygon metric. The memory bank is
built only from hotspot-free training frames. The decision threshold is calibrated on
held-out normal frames, so no validation label is used to choose it. The oracle sweep
over validation thresholds is reported separately and is optimistic by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from torchvision.transforms.functional import gaussian_blur

from solar_ir.datasets.labelme import build_group_split
from solar_ir.metrics.official import f1_tiou
from solar_ir.postprocess.mask import mask_to_polygons

ROOT = Path(__file__).resolve().parents[1]
IR_ROOT = ROOT / "data/raw/datasets/9_16_train_set/train/IR"
GROUPS_CSV = ROOT / "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"
WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
FOLDS = {
    "fold-G01": ["G01", "G06", "G07"],
    "fold-G02": ["G02", "G10"],
    "fold-G03": ["G03", "G05", "G11"],
    "fold-G04": ["G04", "G08", "G09", "G12"],
}
HEIGHT, WIDTH = 512, 640
STRIDE = 8
FEATURE_GRID = (HEIGHT // STRIDE, WIDTH // STRIDE)
PATCHES_PER_IMAGE = FEATURE_GRID[0] * FEATURE_GRID[1]
PROJECTION_DIM = 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def has_hotspot(frame: Path) -> bool:
    annotation = json.loads((IR_ROOT / "label" / f"{frame.stem}.json").read_text(encoding="utf-8"))
    return any(shape.get("label") == "hotspot" for shape in annotation.get("shapes", []))


def hotspot_mask(frame: Path) -> np.ndarray:
    annotation = json.loads((IR_ROOT / "label" / f"{frame.stem}.json").read_text(encoding="utf-8"))
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation.get("shapes", []):
        if shape.get("label") != "hotspot":
            continue
        points = [tuple(float(value) for value in point) for point in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


def load_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    arrays = []
    for path in paths:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if array.shape != (HEIGHT, WIDTH):
            raise ValueError(f"{path.name} has shape {array.shape}, expected {(HEIGHT, WIDTH)}")
        arrays.append(array)
    return torch.from_numpy(np.stack(arrays))[:, None].to(device)


def load_backbone(device: torch.device) -> torchvision.models.ResNet:
    model = torchvision.models.wide_resnet50_2(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval().to(device)


@torch.no_grad()
def patch_features(model: torchvision.models.ResNet, images: torch.Tensor) -> torch.Tensor:
    """Return B x 5120 x 1536 locally aggregated layer2+layer3 descriptors."""
    x = (images.repeat(1, 3, 1, 1) - MEAN.to(images.device)) / STD.to(images.device)
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    mid = model.layer2(model.layer1(x))
    deep = model.layer3(mid)
    mid = F.avg_pool2d(mid, 3, 1, 1)
    deep = F.avg_pool2d(deep, 3, 1, 1)
    deep = F.interpolate(deep, size=mid.shape[-2:], mode="bilinear", align_corners=False)
    features = torch.cat([mid, deep], dim=1)
    return features.flatten(2).transpose(1, 2)


@torch.no_grad()
def greedy_coreset(pool: torch.Tensor, target: int, seed: int) -> torch.Tensor:
    """Farthest-point subsampling in a random low-dimensional projection (PatchCore)."""
    if target > len(pool):
        raise ValueError("coreset target exceeds the candidate pool")
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(pool.shape[1], PROJECTION_DIM, generator=generator)
    projected = pool @ projection.to(pool.device) / math.sqrt(PROJECTION_DIM)
    selected = torch.empty(target, dtype=torch.long, device=pool.device)
    selected[0] = int(torch.randint(len(pool), (1,), generator=generator))
    min_distance = ((projected - projected[selected[0]]) ** 2).sum(dim=1)
    for index in range(1, target):
        selected[index] = torch.argmax(min_distance)
        distance = ((projected - projected[selected[index]]) ** 2).sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)
    return pool[selected]


def build_memory_bank(
    model: torchvision.models.ResNet,
    frames: list[Path],
    device: torch.device,
    *,
    batch_size: int,
    candidate_fraction: float,
    coreset_ratio: float,
    seed: int,
    method: str = "coreset",
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    keep_count = int(PATCHES_PER_IMAGE * candidate_fraction)
    candidates = []
    for start in range(0, len(frames), batch_size):
        features = patch_features(model, load_batch(frames[start : start + batch_size], device))
        for image_features in features:
            keep = torch.randperm(PATCHES_PER_IMAGE, generator=generator)[:keep_count]
            candidates.append(image_features[keep.to(device)])
    pool = torch.cat(candidates)
    target = int(PATCHES_PER_IMAGE * len(frames) * coreset_ratio)
    if method == "random":
        return pool[torch.randperm(len(pool), generator=generator)[:target].to(device)]
    return greedy_coreset(pool, target, seed)


@torch.no_grad()
def anomaly_maps(
    model: torchvision.models.ResNet,
    bank: torch.Tensor,
    frames: list[Path],
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        features = patch_features(model, load_batch(frames[start : start + batch_size], device))
        for image_features in features:
            nearest = torch.cat(
                [
                    torch.cdist(image_features[index : index + 512], bank).min(dim=1).values
                    for index in range(0, PATCHES_PER_IMAGE, 512)
                ]
            )
            score = nearest.view(1, 1, *FEATURE_GRID)
            score = F.interpolate(score, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
            score = gaussian_blur(score, kernel_size=33, sigma=4.0)[0, 0]
            maps.append(score.cpu().numpy())
    return maps


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC; ties are negligible for continuous distance scores."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def evaluate_threshold(
    maps: list[np.ndarray], truths: list[list[Any]], masks: list[np.ndarray], threshold: float
) -> dict[str, float]:
    samples = [
        {
            "ground_truth": {"hotspot": truth},
            "prediction": {"hotspot": mask_to_polygons(score, threshold)},
        }
        for score, truth in zip(maps, truths, strict=True)
    ]
    official = f1_tiou(samples, ("hotspot",))
    intersection = union = 0.0
    for score, mask in zip(maps, masks, strict=True):
        prediction = score >= threshold
        target = mask > 0
        intersection += float((prediction & target).sum())
        union += float((prediction | target).sum())
    return {**official, "pixel_iou": intersection / union if union else 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=[*FOLDS, "photo"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--calib-quantile", type=float, default=0.99)
    parser.add_argument("--candidate-fraction", type=float, default=0.25)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--sweep-points", type=int, default=40)
    parser.add_argument("--bank", choices=["coreset", "random"], default="coreset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/patchcore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir, label_dir = IR_ROOT / "image", IR_ROOT / "label"
    if args.split == "photo":
        train, validation = build_group_split(
            image_dir,
            label_dir,
            GROUPS_CSV,
            0.2,
            split_strategy="photo_stratified",
            seed=args.seed,
        )
    else:
        train, validation = build_group_split(
            image_dir,
            label_dir,
            GROUPS_CSV,
            0.2,
            split_strategy="group",
            seed=args.seed,
            validation_groups=FOLDS[args.split],
        )
    normal_train = sorted(path for path in train if not has_hotspot(path))
    shuffled = normal_train.copy()
    random.Random(args.seed).shuffle(shuffled)
    calib_count = max(1, round(len(shuffled) * args.calib_fraction))
    calib, bank_frames = shuffled[:calib_count], shuffled[calib_count:]
    labels = np.array([has_hotspot(path) for path in validation], dtype=bool)
    print(
        f"[{args.split}] train={len(train)} normal_train={len(normal_train)} "
        f"bank={len(bank_frames)} calib={len(calib)} "
        f"validation={len(validation)} positive={int(labels.sum())}",
        flush=True,
    )

    model = load_backbone(device)
    bank = build_memory_bank(
        model,
        bank_frames,
        device,
        batch_size=args.batch_size,
        candidate_fraction=args.candidate_fraction,
        coreset_ratio=args.coreset_ratio,
        seed=args.seed,
        method=args.bank,
    )
    print(f"[{args.split}] memory bank: {tuple(bank.shape)}", flush=True)

    calib_maps = anomaly_maps(model, bank, calib, device, args.batch_size)
    tau = float(np.quantile([score.max() for score in calib_maps], args.calib_quantile))
    val_maps = anomaly_maps(model, bank, validation, device, args.batch_size)
    val_max = np.array([score.max() for score in val_maps])
    val_masks = [hotspot_mask(path) for path in validation]
    truths = [mask_to_polygons(mask, 0.5) for mask in val_masks]

    calibrated = evaluate_threshold(val_maps, truths, val_masks, tau)
    print(f"[{args.split}] calibrated tau={tau:.4f} -> {calibrated}", flush=True)

    grid = np.linspace(float(min(score.max() for score in calib_maps)), float(val_max.max()), args.sweep_points)
    sweep = []
    for threshold in grid:
        result = evaluate_threshold(val_maps, truths, val_masks, float(threshold))
        sweep.append({"threshold": float(threshold), **result})
    oracle = max(sweep, key=lambda row: row["F1_TIoU"])
    per_image = []
    for path, score, mask in zip(validation, val_maps, val_masks, strict=True):
        prediction = score >= oracle["threshold"]
        target = mask > 0
        per_image.append(
            {
                "frame": path.stem,
                "positive": bool(target.any()),
                "max_score": float(score.max()),
                "gt_area_fraction": float(target.mean()),
                "pred_area_fraction": float(prediction.mean()),
                "pixel_iou_at_oracle": float((prediction & target).sum() / max((prediction | target).sum(), 1)),
            }
        )

    summary: dict[str, Any] = {
        "split": args.split,
        "seed": args.seed,
        "train_frames": len(train),
        "normal_train_frames": len(normal_train),
        "bank_frames": len(bank_frames),
        "calibration_frames": len(calib),
        "validation_frames": len(validation),
        "validation_positive_frames": int(labels.sum()),
        "bank_patches": int(bank.shape[0]),
        "calib_quantile": args.calib_quantile,
        "calibrated_threshold": tau,
        "calibrated": calibrated,
        "image_auroc": auroc(val_max, labels),
        "oracle_threshold": oracle["threshold"],
        "oracle": {key: value for key, value in oracle.items() if key != "threshold"},
        "sweep": sweep,
        "per_image": per_image,
        "bank_method": args.bank,
        "seconds": round(time.time() - started, 1),
    }
    name = args.split if args.bank == "coreset" else f"{args.split}-{args.bank}"
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{args.split}] image_auroc={summary['image_auroc']:.4f} "
        f"calibrated_F1_TIoU={calibrated['F1_TIoU']:.4f} "
        f"oracle_F1_TIoU={oracle['F1_TIoU']:.4f} -> {output_dir / 'metrics.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
