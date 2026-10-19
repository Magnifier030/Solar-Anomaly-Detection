"""Competition task adapter; model architecture remains a configuration plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from research_training.task import TaskBundle
from solar_ir.datasets import ThermalPolygonDataset, build_group_split
from solar_ir.metrics import OfficialLocalizationMetric, dice_score, mean_iou
from solar_ir.models.loader import load_model
from solar_ir.postprocess import mask_to_polygons


class SolarIRTask:
    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        task_config = config["task"]
        self.loss_name = str(config["training"].get("loss", "dice"))
        if self.loss_name not in {"dice", "bce", "bce_dice", "soft_iou"}:
            raise ValueError("training.loss must be dice, bce, bce_dice or soft_iou")
        self.metric_threshold = float(task_config.get("metric_threshold", 0.5))
        self.localization_metric = OfficialLocalizationMetric(("hotspot",))
        dataset_root = Path(str(task_config.get("dataset_root", "data/raw/datasets/9_16_train_set")))
        image_dir = dataset_root / "train/IR/image"
        label_dir = dataset_root / "train/IR/label"
        groups = Path(str(task_config.get("groups_csv", "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv")))
        split_strategy = str(task_config.get("split_strategy", "group"))
        train_paths, validation_paths = build_group_split(
            image_dir,
            label_dir,
            groups,
            float(task_config.get("validation_fraction", 0.2)),
            split_strategy=split_strategy,
            seed=int(config["training"].get("seed", 42)),
        )
        raw_image_size = task_config.get("image_size", 256)
        if isinstance(raw_image_size, (list, tuple)):
            image_size: int | tuple[int, int] = tuple(int(value) for value in raw_image_size)
            if len(image_size) != 2:
                raise ValueError("task.image_size list must contain [width, height]")
        else:
            image_size = int(raw_image_size)
        augmentation = dict(task_config.get("augmentation", {}))
        train_dataset = ThermalPolygonDataset(
            train_paths,
            label_dir,
            image_size,
            augment=bool(augmentation.get("enabled", False)),
            augmentation=augmentation,
        )
        validation_dataset = ThermalPolygonDataset(validation_paths, label_dir, image_size)
        generator = torch.Generator().manual_seed(int(config["training"].get("seed", 42)))
        train_loader = DataLoader(train_dataset, batch_size=int(config["training"].get("batch_size", 8)), shuffle=True, generator=generator, num_workers=0)
        validation_loader = DataLoader(validation_dataset, batch_size=int(config["training"].get("batch_size", 8)), shuffle=False, num_workers=0)
        model = load_model(dict(task_config["model"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-4)))
        return TaskBundle(model, optimizer, train_loader, validation_loader, {
            "modality": "IR", "anomaly_label": "hotspot", "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset), "split": split_strategy,
            "loss": self.loss_name,
        }, generator)

    def training_loss(self, model: nn.Module, batch: Any, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        images, masks = (value.to(device) for value in batch)
        raw_output = model(images)
        logits_list = raw_output if isinstance(raw_output, list) else [raw_output]
        logits = logits_list[-1]
        bce_values = [
            nn.functional.binary_cross_entropy_with_logits(prediction, masks)
            for prediction in logits_list
        ]
        dice_values = [self._soft_dice_loss(prediction, masks) for prediction in logits_list]
        soft_iou_values = [self._soft_iou_loss(prediction, masks) for prediction in logits_list]
        if self.loss_name == "dice":
            loss = torch.stack(dice_values).mean()
        elif self.loss_name == "bce":
            loss = torch.stack(bce_values).mean()
        elif self.loss_name == "bce_dice":
            loss = torch.stack(bce_values).mean() + torch.stack(dice_values).mean()
        else:
            loss = torch.stack(soft_iou_values).mean()
        return loss, {
            "bce": float(bce_values[-1].detach()),
            "dice_loss": float(dice_values[-1].detach()),
            "soft_iou_loss": float(soft_iou_values[-1].detach()),
        }

    @staticmethod
    def _soft_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * masks).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + masks.sum(dim=dimensions)
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        return (1.0 - dice).mean()

    @staticmethod
    def _soft_iou_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * masks).sum(dim=dimensions)
        union = probabilities.sum(dim=dimensions) + masks.sum(dim=dimensions) - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        return (1.0 - iou).mean()

    @torch.no_grad()
    def validation_metrics(self, model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, float]:
        model.eval()
        ious: list[float] = []
        dices: list[float] = []
        localization_samples: list[dict[str, Any]] = []
        for images, masks in loader:
            raw_output = model(images.to(device))
            logits = raw_output[-1] if isinstance(raw_output, list) else raw_output
            ious.append(mean_iou(logits.cpu(), masks))
            dices.append(dice_score(logits.cpu(), masks))
            probabilities = torch.sigmoid(logits).cpu()
            for probability, target in zip(probabilities, masks):
                localization_samples.append({
                    "ground_truth": {"hotspot": mask_to_polygons(target[0].numpy(), self.metric_threshold)},
                    "prediction": {"hotspot": mask_to_polygons(probability[0].numpy(), self.metric_threshold)},
                })
        official = self.localization_metric(localization_samples)
        return {
            "mean_iou": sum(ious) / len(ious),
            "dice": sum(dices) / len(dices),
            **official,
        }


def create_task() -> SolarIRTask:
    return SolarIRTask()
