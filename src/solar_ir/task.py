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
        self.metric_threshold = float(task_config.get("metric_threshold", 0.5))
        self.localization_metric = OfficialLocalizationMetric(("hotspot",))
        dataset_root = Path(str(task_config.get("dataset_root", "data/raw/datasets/9_16_train_set")))
        image_dir = dataset_root / "train/IR/image"
        label_dir = dataset_root / "train/IR/label"
        groups = Path(str(task_config.get("groups_csv", "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv")))
        split_strategy = str(task_config.get("split_strategy", "random_stratified"))
        train_paths, validation_paths = build_group_split(
            image_dir,
            label_dir,
            groups,
            float(task_config.get("validation_fraction", 0.2)),
            split_strategy=split_strategy,
            seed=int(config["training"].get("seed", 42)),
        )
        image_size = int(task_config.get("image_size", 256))
        train_dataset = ThermalPolygonDataset(train_paths, label_dir, image_size)
        validation_dataset = ThermalPolygonDataset(validation_paths, label_dir, image_size)
        generator = torch.Generator().manual_seed(int(config["training"].get("seed", 42)))
        train_loader = DataLoader(train_dataset, batch_size=int(config["training"].get("batch_size", 8)), shuffle=True, generator=generator, num_workers=0)
        validation_loader = DataLoader(validation_dataset, batch_size=int(config["training"].get("batch_size", 8)), shuffle=False, num_workers=0)
        model = load_model(dict(task_config["model"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-4)))
        return TaskBundle(model, optimizer, train_loader, validation_loader, {
            "modality": "IR", "anomaly_label": "hotspot", "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset), "split": split_strategy,
        }, generator)

    def training_loss(self, model: nn.Module, batch: Any, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        images, masks = (value.to(device) for value in batch)
        raw_output = model(images)
        logits_list = raw_output if isinstance(raw_output, list) else [raw_output]
        logits = logits_list[-1]
        bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * masks).sum()
        dice_loss = 1 - (2 * intersection + 1) / (probabilities.sum() + masks.sum() + 1)
        deep_loss = sum(
            nn.functional.binary_cross_entropy_with_logits(prediction, masks)
            for prediction in logits_list
        ) / len(logits_list)
        return deep_loss + dice_loss, {"bce": float(bce.detach()), "dice_loss": float(dice_loss.detach())}

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
