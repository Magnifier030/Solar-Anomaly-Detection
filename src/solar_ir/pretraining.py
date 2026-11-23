"""DNA-Net pretraining on explicitly weak bbox masks, separate from competition metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from research_training.progress import BatchProgress, report
from research_training.task import TaskBundle
from solar_ir.datasets.masks import ThermalMaskDataset
from solar_ir.models.loader import load_model
from solar_ir.optimization import build_optimizer
from solar_ir.preprocessing.pvhsd2025 import file_hash
from solar_ir.task import SolarIRTask


class PVHSDPretrainingTask(SolarIRTask):
    def build(self, config: dict[str, Any], device: torch.device) -> TaskBundle:
        task, training = config["task"], config["training"]
        self.loss_name = str(training.get("loss", "soft_iou"))
        if self.loss_name != "soft_iou":
            raise ValueError("This DNA-Net pretraining recipe requires official soft_iou loss")
        if int(task["model"].get("input_channels", 1)) != 1:
            raise ValueError("Prepared PV-HSD images require input_channels=1")
        root = Path(task["dataset_root"])
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("supervision") != "bbox_raster_weak_mask"
            or manifest["input_channels"] != 1
        ):
            raise ValueError("Expected one-channel bbox-derived weak-mask dataset")
        raw_size = task.get("image_size", [640, 512])
        size = (raw_size, raw_size) if isinstance(raw_size, int) else tuple(raw_size)
        if len(size) != 2 or any(int(n) % 16 for n in size):
            raise ValueError("DNANet width and height must be multiples of 16")
        augmentation = dict(task.get("augmentation", {}))
        samples = manifest["samples"]
        families = {
            split: {r["family_hint"] for r in samples if r["split"] == split}
            for split in ["train", "validation"]
        }
        if families["train"] & families["validation"]:
            raise ValueError("Filename families overlap between pretraining splits")
        report(f"PV-HSD: verifying {len(samples)} image/mask pairs (SHA-256)...")
        for sample_index, record in enumerate(samples, 1):
            for key in ["image", "mask"]:
                path = root / record[key]
                if root.resolve() not in path.resolve().parents:
                    raise ValueError("Manifest path escapes dataset directory")
                if file_hash(path) != record[f"{key}_sha256"]:
                    raise ValueError(f"Prepared dataset changed: {path}")
            if sample_index == 1 or sample_index % 500 == 0 or sample_index == len(samples):
                report(f"PV-HSD integrity check: {sample_index}/{len(samples)} pairs")
        datasets = {}
        for split in ["train", "validation"]:
            paths = [root / r["image"] for r in samples if r["split"] == split]
            if not paths:
                raise ValueError(f"Empty {split} dataset")
            datasets[split] = ThermalMaskDataset(
                paths,
                root / "masks",
                (int(size[0]), int(size[1])),
                augment=split == "train" and bool(augmentation.get("enabled", False)),
                augmentation=augmentation,
                preprocessing="grayscale",
            )
        generator = torch.Generator().manual_seed(int(training.get("seed", 7)))
        batch_size = int(training.get("batch_size", 4))
        drop_last = bool(training.get("drop_last", True))
        if drop_last and len(datasets["train"]) < batch_size:
            raise ValueError("batch_size with drop_last would produce an empty training loader")
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=drop_last,
        )
        validation_loader = DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0
        )
        report(
            f"PV-HSD: building model | train={len(datasets['train'])} | "
            f"validation={len(datasets['validation'])} | batch_size={batch_size}"
        )
        model = load_model(dict(task["model"])).to(device)
        optimizer = build_optimizer(model, training)
        return TaskBundle(
            model,
            optimizer,
            train_loader,
            validation_loader,
            {
                "dataset": "PV-HSD-2025",
                "external_dataset": True,
                "source_revision": manifest["signature"]["source_revision"],
                "source_archive_sha256": manifest["source_archive_sha256"],
                "prepared_manifest_sha256": file_hash(manifest_path),
                "supervision": "bbox_raster_weak_mask",
                "input_channels": 1,
                "train_samples": len(datasets["train"]),
                "validation_samples": len(datasets["validation"]),
                "split_policy": manifest["split_policy"],
                "loss": self.loss_name,
                "metrics_scope": "agreement with bbox weak masks, not competition F1_TIoU",
            },
            generator,
        )

    @torch.no_grad()
    def validation_metrics(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        model.eval()
        intersection = union = total = 0
        progress = BatchProgress("Validation", len(loader))
        for batch_index, (images, masks) in enumerate(loader, 1):
            outputs = model(images.to(device))
            logits = outputs[-1] if isinstance(outputs, list) else outputs
            prediction, target = logits.cpu() > 0, masks > 0.5
            intersection += int((prediction & target).sum())
            union += int((prediction | target).sum())
            total += int(prediction.sum() + target.sum())
            progress.update(batch_index)
        return {
            "weak_mask_iou": intersection / (union + 2.220446049250313e-16),
            "weak_mask_dice": (2 * intersection + 1e-6) / (total + 1e-6),
        }


def create_task() -> PVHSDPretrainingTask:
    return PVHSDPretrainingTask()
