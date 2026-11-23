"""Module instance segmentation with Mask R-CNN, one model per modality (IR or EO).

The competition scores whole modules: ``solar_panel`` for normal modules and
``hotspot`` (IR) / ``crack`` (EO) for anomalous ones, each as one polygon. IR and EO
frames of the same name are scored as independent images in different coordinate
systems, so each modality gets its own detector with two foreground classes
(normal module, anomalous module). Torchvision's COCO-pretrained Mask R-CNN v2 is
fine-tuned on the training groups; the held-out groups are never used for training,
checkpoint selection or thresholds.

IR polarity flips between groups (white-hot and black-hot), so IR training applies a
random intensity inversion. Pseudo-colour IR frames (test set) are mapped back to a
grey level through the nearest entry of the inferno colour map.

Usage::

    python scripts/module_maskrcnn.py train --modality ir --output outputs/module-maskrcnn-ir-g013
    python scripts/module_maskrcnn.py evaluate --run outputs/module-maskrcnn-ir-g013 \\
        --run outputs/module-maskrcnn-eo-g013
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from solar_ir.metrics.official import f1_tiou

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/datasets/9_16_train_set/train"
GROUPS = ROOT / "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"
ANOMALY = {"ir": "hotspot", "eo": "crack"}
FOLDER = {"ir": "IR", "eo": "EO"}
# Detector input scale: IR 640x512 is upsampled 2x (G06/G08 modules are ~20 px wide).
SIZES = {"ir": (1024, 1280), "eo": (1080, 1920)}
INFERNO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[
    :, 0
].astype(np.float32)


def frames_by_group() -> dict[str, str]:
    with GROUPS.open() as handle:
        return {row["frame_id"].split(".")[0]: row["group"] for row in csv.DictReader(handle)}


def split(validation_groups: list[str]) -> tuple[list[str], list[str]]:
    groups = frames_by_group()
    names = sorted(groups)
    held_out = set(validation_groups)
    return [n for n in names if groups[n] not in held_out], [n for n in names if groups[n] in held_out]


def to_grey(image: np.ndarray) -> np.ndarray:
    """BGR frame → single grey plane; pseudo-colour frames via the inferno colour map."""
    if np.abs(image[..., 0].astype(np.int16) - image[..., 2]).mean() <= 5:
        return image[..., 1]
    pixels = image.reshape(-1, 3).astype(np.float32)
    index = np.empty(len(pixels), dtype=np.uint8)
    for start in range(0, len(pixels), 65536):
        chunk = pixels[start : start + 65536]
        index[start : start + 65536] = ((chunk[:, None] - INFERNO[None]) ** 2).sum(-1).argmin(1)
    return index.reshape(image.shape[:2])


def load_image(path: Path, modality: str) -> np.ndarray:
    """(H, W, 3) uint8 RGB-ordered network input."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if modality == "ir":
        grey = to_grey(image)
        return np.repeat(grey[..., None], 3, axis=2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_path(folder: Path, name: str) -> Path:
    return next(folder.glob(f"{name}.*"))


def load_shapes(name: str, modality: str) -> list[tuple[str, np.ndarray]]:
    data = json.loads((TRAIN / FOLDER[modality] / "label" / f"{name}.json").read_text())
    return [(s["label"], np.asarray(s["points"], dtype=np.float32)) for s in data["shapes"]]


class ModuleDataset(torch.utils.data.Dataset[Any]):
    def __init__(self, names: list[str], modality: str, augment: bool) -> None:
        self.names, self.modality, self.augment = names, modality, augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        name = self.names[index]
        image = load_image(image_path(TRAIN / FOLDER[self.modality] / "image", name), self.modality)
        height, width = image.shape[:2]
        masks, labels = [], []
        for label, points in load_shapes(name, self.modality):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
            if mask.sum() < 4:
                continue
            masks.append(mask)
            labels.append(2 if label == ANOMALY[self.modality] else 1)
        stack = np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
        if self.augment:
            if random.random() < 0.5:
                image, stack = image[:, ::-1], stack[:, :, ::-1]
            if random.random() < 0.5:
                image, stack = image[::-1], stack[:, ::-1]
            if self.modality == "ir" and random.random() < 0.5:
                image = 255 - image
            gain, bias = random.uniform(0.7, 1.3), random.uniform(-25, 25)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        stack = np.ascontiguousarray(stack)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(stack)
        boxes = []
        for mask in stack:
            ys, xs = np.nonzero(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": mask_tensor,
        }
        return tensor, target


def build_model(modality: str, pretrained: bool = True) -> torch.nn.Module:
    min_size, max_size = SIZES[modality]
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=300,
        rpn_pre_nms_top_n_test=3000,
        rpn_post_nms_top_n_test=2000,
        rpn_pre_nms_top_n_train=3000,
        rpn_post_nms_top_n_train=2000,
        box_batch_size_per_image=512,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 3)
    channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(channels, 256, 3)
    return model


def collate(batch: list[Any]) -> tuple[Any, ...]:
    return tuple(zip(*batch, strict=True))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    train_names, validation_names = split(args.validation_groups)
    loader = torch.utils.data.DataLoader(
        ModuleDataset(train_names, args.modality, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_model(args.modality).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    warmup = min(500, total // 10)

    def schedule(step: int) -> float:
        if step < warmup:
            return 0.001 + (1 - 0.001) * step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "train_frames": len(train_names), "validation_frames": len(validation_names)}
    (output / "config.json").write_text(json.dumps(settings, indent=1))
    log = (output / "train.log").open("a")
    step, started = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                parts = " ".join(f"{k} {float(v):.3f}" for k, v in losses.items())
                line = (
                    f"epoch {epoch} step {step}/{total} loss {float(loss):.3f} {parts} "
                    f"lr {scheduler.get_last_lr()[0]:.5f} {time.time() - started:.0f}s"
                )
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": settings}, output / "last.pt")
    print("done", flush=True)


def mask_polygon(mask: np.ndarray) -> list[list[float]] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    epsilon = 1.0
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > 100:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    return [[float(x), float(y)] for x, y in approx[:, 0]]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


@torch.no_grad()
def predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    """Every detection: label (1 normal, 2 anomaly), score, polygon (original pixels)."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)
    result = model([tensor])[0]
    masks = (result["masks"][:, 0] > 0.5).cpu().numpy()
    detections = []
    for mask, label, score in zip(masks, result["labels"].tolist(), result["scores"].tolist(), strict=True):
        polygon = mask_polygon(mask)
        if polygon is not None:
            detections.append({"label": int(label), "score": float(score), "polygon": polygon, "mask": mask})
    return detections


def select(
    detections: list[dict[str, Any]], threshold: float, anomaly_threshold: float, overlap: float
) -> list[dict[str, Any]]:
    """Score thresholds per class, then class-agnostic mask NMS (one label per module)."""
    kept: list[dict[str, Any]] = []
    candidates = [
        d
        for d in detections
        if d["score"] >= (anomaly_threshold if d["label"] == 2 else threshold)
    ]
    for detection in sorted(candidates, key=lambda d: -d["score"]):
        if all(mask_iou(detection["mask"], k["mask"]) < overlap for k in kept):
            kept.append(detection)
    return kept


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    report: dict[str, Any] = {}
    pooled: dict[str, list[dict[str, Any]]] = {}
    for run in args.run:
        config = json.loads((Path(run) / "config.json").read_text())
        modality = config["modality"]
        model = build_model(modality, pretrained=False)
        model.load_state_dict(torch.load(Path(run) / "last.pt", map_location="cpu")["model"])
        model.to(device).eval()
        _, validation_names = split(config["validation_groups"])
        groups = frames_by_group()
        raw = []
        for name in validation_names:
            image = load_image(image_path(TRAIN / FOLDER[modality] / "image", name), modality)
            detections = predict(model, image, device)
            truth: dict[str, list[Any]] = {}
            for label, points in load_shapes(name, modality):
                truth.setdefault(label, []).append(points.tolist())
            raw.append({"name": name, "group": groups[name], "truth": truth, "detections": detections})
        for threshold in args.thresholds:
            for anomaly in args.anomaly_thresholds:
                samples = []
                for item in raw:
                    prediction: dict[str, list[Any]] = {}
                    for d in select(item["detections"], threshold, anomaly, args.overlap):
                        label = ANOMALY[modality] if d["label"] == 2 else "solar_panel"
                        prediction.setdefault(label, []).append(d["polygon"])
                    samples.append({"ground_truth": item["truth"], "prediction": prediction, "group": item["group"]})
                key = f"{threshold:.2f}/{anomaly:.2f}"
                pooled.setdefault(key, []).extend(samples)
                result = {"all": f1_tiou(samples)}
                for group in sorted({s["group"] for s in samples}):
                    result[group] = f1_tiou([s for s in samples if s["group"] == group])
                report.setdefault(Path(run).name, {})[key] = result
                print(Path(run).name, key, json.dumps(result["all"]), flush=True)
        dump = [
            {
                "name": item["name"],
                "group": item["group"],
                "truth": item["truth"],
                "detections": [{k: v for k, v in d.items() if k != "mask"} for d in item["detections"]],
            }
            for item in raw
        ]
        (Path(run) / "validation_detections.json").write_text(json.dumps(dump))
    if len(args.run) > 1:
        report["pooled"] = {key: f1_tiou(samples) for key, samples in pooled.items()}
        for key, value in report["pooled"].items():
            print("pooled", key, json.dumps(value), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--modality", choices=["ir", "eo"], required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--validation-groups", nargs="+", default=["G01", "G03", "G07"])
    training.add_argument("--epochs", type=int, default=24)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=0.01)
    training.add_argument("--seed", type=int, default=7)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", action="append", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    evaluation.add_argument("--anomaly-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    evaluation.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
