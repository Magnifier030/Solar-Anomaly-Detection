"""Competition task adapter; model architecture remains a configuration plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from research_training.progress import BatchProgress, report
from research_training.task import TaskBundle
from solar_ir.datasets import ThermalPolygonDataset, build_group_split
from solar_ir.datasets.labelme import _has_hotspot
from solar_ir.finetuning import apply_finetuning_policy
from solar_ir.metrics import OfficialLocalizationMetric, dice_score, mean_iou
from solar_ir.models.loader import load_model
from solar_ir.optimization import build_optimizer
from solar_ir.postprocess import mask_to_polygons


class SolarIRTask:
    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        task_config = config["task"]
        self.loss_name = str(config["training"].get("loss", "dice"))
        if self.loss_name not in {"dice", "bce", "bce_dice", "soft_iou", "inp"}:
            raise ValueError("training.loss must be dice, bce, bce_dice, soft_iou or inp")
        self.metric_threshold = float(task_config.get("metric_threshold", 0.5))
        self.metric_aggregation = str(task_config.get("metric_aggregation", "batch_mean"))
        if self.metric_aggregation not in {"batch_mean", "global"}:
            raise ValueError("metric_aggregation must be batch_mean or global")
        self.localization_metric = OfficialLocalizationMetric(("hotspot",))
        dataset_root = Path(
            str(task_config.get("dataset_root", "data/raw/datasets/9_16_train_set"))
        )
        image_dir = dataset_root / "train/IR/image"
        label_dir = dataset_root / "train/IR/label"
        groups = Path(
            str(
                task_config.get(
                    "groups_csv",
                    "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv",
                )
            )
        )
        split_strategy = str(task_config.get("split_strategy", "group"))
        report(f"Solar IR: reading annotations and building {split_strategy} split...")
        if split_strategy == "photo_stratified":
            report(
                "Photo-level development split: groups may overlap; "
                "physical panel identity is not verified."
            )
        train_paths, validation_paths = build_group_split(
            image_dir,
            label_dir,
            groups,
            float(task_config.get("validation_fraction", 0.2)),
            split_strategy=split_strategy,
            seed=int(config["training"].get("seed", 42)),
            validation_groups=task_config.get("validation_groups"),
        )
        if task_config.get("train_normal_only", False):
            # Unsupervised objectives (e.g. INP-Former) learn only from hotspot-free frames.
            train_paths = [path for path in train_paths if not _has_hotspot(label_dir, path.stem)]
        raw_image_size = task_config.get("image_size", 256)
        if isinstance(raw_image_size, (list, tuple)):
            if len(raw_image_size) != 2:
                raise ValueError("task.image_size list must contain [width, height]")
            image_size: int | tuple[int, int] = (int(raw_image_size[0]), int(raw_image_size[1]))
        else:
            image_size = int(raw_image_size)
        preprocessing = str(task_config.get("preprocessing", "grayscale"))
        expected_channels = 3 if preprocessing == "dnanet" else 1
        if int(task_config["model"].get("input_channels", expected_channels)) != expected_channels:
            raise ValueError("model.input_channels does not match task.preprocessing")
        augmentation = dict(task_config.get("augmentation", {}))
        train_dataset = ThermalPolygonDataset(
            train_paths,
            label_dir,
            image_size,
            augment=bool(augmentation.get("enabled", False)),
            augmentation=augmentation,
            preprocessing=str(task_config.get("preprocessing", "grayscale")),
        )
        validation_dataset = ThermalPolygonDataset(
            validation_paths,
            label_dir,
            image_size,
            preprocessing=str(task_config.get("preprocessing", "grayscale")),
        )
        generator = torch.Generator().manual_seed(int(config["training"].get("seed", 42)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"].get("batch_size", 8)),
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=bool(config["training"].get("drop_last", False)),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["training"].get("batch_size", 8)),
            shuffle=False,
            num_workers=0,
        )
        report(
            f"Solar IR: building model | train={len(train_dataset)} | "
            f"validation={len(validation_dataset)}"
        )
        model = load_model(dict(task_config["model"])).to(device)
        training = config["training"]
        finetuning = apply_finetuning_policy(model, training.get("finetuning", {}))
        report(
            f"Finetuning: {finetuning['mode']} | "
            f"trainable={finetuning['trainable_parameters']} | "
            f"frozen={finetuning['frozen_parameters']} | "
            f"modules={finetuning['frozen_modules']} | "
            f"frozen running stats={finetuning['frozen_running_stats']}"
        )
        optimizer = build_optimizer(model, training)
        optimizer_name = str(training.get("optimizer", "AdamW"))
        schedule = str(training.get("scheduler", "none"))
        return TaskBundle(
            model,
            optimizer,
            train_loader,
            validation_loader,
            {
                "modality": "IR",
                "anomaly_label": "hotspot",
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "split": split_strategy,
                "loss": self.loss_name,
                "train_frames": [path.stem for path in train_paths],
                "validation_frames": [path.stem for path in validation_paths],
                "optimizer": optimizer_name,
                "finetuning": finetuning,
                "scheduler": schedule,
                "metric_aggregation": self.metric_aggregation,
                "external_pretraining": task_config.get("external_pretraining"),
            },
            generator,
        )

    def training_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        images, masks = (value.to(device) for value in batch)
        if self.loss_name == "inp":
            # Unsupervised objective: masks are not read during training.
            return model.objective(images)
        raw_output = model(images)
        logits_list = raw_output if isinstance(raw_output, list) else [raw_output]
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
        # Official SoftIoULoss reduces the WHOLE batch, not each image separately.
        intersection = (probabilities * masks).sum()
        union = probabilities.sum() + masks.sum() - intersection
        return 1.0 - (intersection + 1.0) / (union + 1.0)

    @torch.no_grad()
    def validation_metrics(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        model.eval()
        ious: list[float] = []
        dices: list[float] = []
        localization_samples: list[dict[str, Any]] = []
        intersection = union = total = 0
        progress = BatchProgress("Validation", len(loader))
        for batch_index, (images, masks) in enumerate(loader, 1):
            raw_output = model(images.to(device))
            logits = raw_output[-1] if isinstance(raw_output, list) else raw_output
            ious.append(mean_iou(logits.cpu(), masks))
            dices.append(dice_score(logits.cpu(), masks))
            prediction = logits.cpu() > 0
            target_mask = masks > 0
            intersection += int((prediction & target_mask).sum())
            union += int((prediction | target_mask).sum())
            total += int(prediction.sum() + target_mask.sum())
            probabilities = torch.sigmoid(logits).cpu()
            for probability, target in zip(probabilities, masks, strict=True):
                localization_samples.append(
                    {
                        "ground_truth": {
                            "hotspot": mask_to_polygons(target[0].numpy(), self.metric_threshold)
                        },
                        "prediction": {
                            "hotspot": mask_to_polygons(
                                probability[0].numpy(), self.metric_threshold
                            )
                        },
                    }
                )
            progress.update(batch_index)
        report("Validation: aggregating official polygon metrics...")
        official = self.localization_metric(localization_samples)
        return {
            "mean_iou": (
                intersection / (union + 2.220446049250313e-16)
                if self.metric_aggregation == "global"
                else sum(ious) / len(ious)
            ),
            "dice": (
                (2 * intersection + 1e-6) / (total + 1e-6)
                if self.metric_aggregation == "global"
                else sum(dices) / len(dices)
            ),
            **official,
        }


def create_task() -> SolarIRTask:
    return SolarIRTask()
