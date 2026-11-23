"""Three-target IR/RGB segmentation task: solar_panel, hotspot and crack.

The input modality is a configuration choice (ir, rgb or rgbir), so IR-only, RGB-only
and four-channel fusion runs share the same split, model, loss and metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from research_training.progress import BatchProgress, report
from research_training.task import TaskBundle
from solar_ir.datasets import build_group_split
from solar_ir.datasets.paired import TARGET_CLASSES, PairedSolarDataset, channel_count
from solar_ir.metrics import OfficialLocalizationMetric
from solar_ir.models.loader import load_model
from solar_ir.postprocess import mask_to_polygons


class FusionSegmentationTask:
    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        task_config = config["task"]
        training = config["training"]
        modalities = str(task_config["modalities"])
        canvas = int(task_config.get("canvas", 672))
        normalize = str(task_config.get("normalize", "none"))
        self.threshold = float(task_config.get("metric_threshold", 0.5))
        # Per-class loss weights (solar_panel, hotspot, crack). Panels cover most pixels and
        # swamp the sparse classes, so the weights let the rarer classes contribute.
        self.class_weights = torch.tensor(
            task_config.get("class_weights", [1.0, 1.0, 1.0]), dtype=torch.float32
        ).view(1, -1, 1, 1)
        root = Path(str(task_config["dataset_root"]))
        ir_image_dir = root / "train/IR/image"
        ir_label_dir = root / "train/IR/label"
        eo_image_dir = root / "train/EO/image"
        eo_label_dir = root / "train/EO/label"
        train_paths, validation_paths = build_group_split(
            ir_image_dir,
            ir_label_dir,
            Path(str(task_config["groups_csv"])),
            float(task_config.get("validation_fraction", 0.2)),
            split_strategy=str(task_config.get("split_strategy", "group")),
            seed=int(training.get("seed", 42)),
            validation_groups=task_config.get("validation_groups"),
        )
        report(
            f"Fusion: modalities={modalities} canvas={canvas} | "
            f"train={len(train_paths)} | validation={len(validation_paths)}"
        )
        photometric = bool(task_config.get("photometric_augment", False))
        invert_probability = float(task_config.get("invert_probability", 0.0))
        module_boundary = bool(task_config.get("module_boundary", False))
        train_dataset = PairedSolarDataset(
            train_paths,
            ir_label_dir,
            eo_image_dir,
            eo_label_dir,
            canvas,
            modalities,
            normalize,
            photometric,
            invert_probability,
            module_boundary=module_boundary,
        )
        validation_dataset = PairedSolarDataset(
            validation_paths,
            ir_label_dir,
            eo_image_dir,
            eo_label_dir,
            canvas,
            modalities,
            normalize,
        )
        pseudo_dir = task_config.get("pseudo_label_dir")
        if pseudo_dir:
            # Self-training: unlabeled target frames join training with pseudo-labels only.
            pseudo_dataset = PairedSolarDataset(
                validation_paths,
                ir_label_dir,
                eo_image_dir,
                eo_label_dir,
                canvas,
                modalities,
                normalize,
                photometric,
                invert_probability,
                target_dir=Path(str(pseudo_dir)),
            )
            train_dataset = ConcatDataset([train_dataset, pseudo_dataset])
        batch_size = int(training.get("batch_size", 4))
        generator = torch.Generator().manual_seed(int(training.get("seed", 42)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=bool(training.get("drop_last", False)),
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        model_config = dict(task_config["model"])
        model_config["channels"] = channel_count(modalities)
        model = load_model(model_config).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]))
        report(
            f"Fusion: trainable parameters={sum(p.numel() for p in trainable)} "
            f"| input channels={model_config['channels']}"
        )
        return TaskBundle(
            model,
            optimizer,
            train_loader,
            validation_loader,
            {
                "modalities": modalities,
                "channels": model_config["channels"],
                "canvas": canvas,
                "target_classes": list(TARGET_CLASSES),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_frames": [path.stem for path in train_paths],
                "validation_frames": [path.stem for path in validation_paths],
                "trainable_parameters": sum(p.numel() for p in trainable),
            },
            generator,
        )

    def training_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        images, targets = (value.to(device) for value in batch)
        logits = model(images)
        weights = self.class_weights.to(device)
        bce_map = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (bce_map.mean(dim=(0, 2, 3)) * weights.flatten()).sum() / weights.sum()
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(2, 3))
        denominator = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        dice_loss = ((1.0 - dice).mean(dim=0) * weights.flatten()).sum() / weights.sum()
        return bce + dice_loss, {"bce": float(bce.detach()), "dice_loss": float(dice_loss.detach())}

    @torch.no_grad()
    def validation_metrics(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        model.eval()
        samples: list[dict[str, Any]] = []
        intersection = dict.fromkeys(TARGET_CLASSES, 0.0)
        union = dict.fromkeys(TARGET_CLASSES, 0.0)
        progress = BatchProgress("Validation", len(loader))
        for batch_index, (images, targets) in enumerate(loader, 1):
            probabilities = torch.sigmoid(model(images.to(device))).cpu()
            for probability, target in zip(probabilities, targets, strict=True):
                ground_truth: dict[str, list[Any]] = {}
                prediction: dict[str, list[Any]] = {}
                for index, class_name in enumerate(TARGET_CLASSES):
                    truth_mask = target[index].numpy()
                    predicted = probability[index].numpy()
                    ground_truth[class_name] = mask_to_polygons(truth_mask, 0.5)
                    prediction[class_name] = mask_to_polygons(predicted, self.threshold)
                    hit = (predicted >= self.threshold) & (truth_mask > 0.5)
                    either = (predicted >= self.threshold) | (truth_mask > 0.5)
                    intersection[class_name] += float(hit.sum())
                    union[class_name] += float(either.sum())
                samples.append({"ground_truth": ground_truth, "prediction": prediction})
            progress.update(batch_index)
        report("Validation: aggregating official polygon metrics...")
        metric = OfficialLocalizationMetric(TARGET_CLASSES)
        official = metric(samples)
        pixel_iou = {
            f"pixel_iou_{name}": intersection[name] / (union[name] + 1e-9)
            for name in TARGET_CLASSES
        }
        return {**official, **pixel_iou}


def create_task() -> FusionSegmentationTask:
    return FusionSegmentationTask()
