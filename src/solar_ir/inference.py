"""Build an IR-only competition submission from a saved training run."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from research_training.backends import configure_backends
from research_training.storage import file_sha256
from solar_ir.datasets.labelme import ThermalPolygonDataset
from solar_ir.models.loader import load_model
from solar_ir.postprocess.submission import submission_polygons


class TestImages(Dataset):
    """Use the training validation transform, without opening any annotations."""

    def __init__(self, keys: list[str], root: Path, task: dict) -> None:
        self.keys, self.root = keys, root
        self.transform = ThermalPolygonDataset(
            [], Path("."), tuple(task["image_size"]), preprocessing=task["preprocessing"]
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        with Image.open(self.root / key) as source:
            image = source.convert("L")
            width, height = image.size
            tensor, _ = self.transform._prepare_tensors(image, Image.new("L", image.size))
        return tensor, key, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(settings["training_config"]).read_text())
    task = config["task"]
    if task["preprocessing"] != "grayscale":
        raise ValueError("This IR submission pipeline requires grayscale preprocessing")
    root = Path(settings["test_root"])
    kit = Path(settings["evaluation_kit"])
    with (kit / "test_images.csv").open(encoding="utf-8-sig", newline="") as stream:
        keys = [row["filename"] for row in csv.DictReader(stream)]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate test keys")
    ir_keys = [key for key in keys if key.startswith("test/IR/image/")]
    if not ir_keys or any(not key.startswith(("test/IR/image/", "test/EO/image/")) for key in keys):
        raise ValueError("Unexpected test manifest")
    for key in keys:
        if not (root / key).is_file():
            raise FileNotFoundError(root / key)
    device = torch.device(settings["device"])
    backend = configure_backends(config, device)
    model_config = copy.deepcopy(task["model"])
    model_config.pop("pretrained_path", None)
    model = load_model(model_config).to(device).eval()
    checkpoint_path = Path(settings["checkpoint"])
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != json.loads(json.dumps(config)):
        # Resolved training configs may additionally contain artifact hashes.
        saved_model = checkpoint.get("config", {}).get("task", {}).get("model")
        if saved_model != task["model"]:
            raise ValueError("Checkpoint architecture differs from training config")
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = DataLoader(TestImages(ir_keys, root, task), batch_size=settings["batch_size"])
    submission = dict.fromkeys(keys)
    submission = {key: [] for key in submission}
    mask_dir = output / "ir_masks"
    mask_dir.mkdir()
    per_image = []
    for batch_index, (images, batch_keys, widths, heights) in enumerate(loader, 1):
        with torch.inference_mode():
            raw = model(images.to(device))
            logits = raw[-1] if isinstance(raw, list) else raw
            probabilities = logits.sigmoid().cpu()
        for probability, key, width, height in zip(
            probabilities, batch_keys, widths, heights, strict=True
        ):
            native = torch.nn.functional.interpolate(
                probability.unsqueeze(0),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            mask = (native.numpy() >= settings["threshold"]).astype(np.uint8) * 255
            polygons = submission_polygons(mask)
            if len(polygons) > 500:
                raise ValueError(f"Too many objects in {key}: {len(polygons)}")
            submission[key] = [
                {"label": "hotspot", "points": points, "reason": None} for points in polygons
            ]
            Image.fromarray(mask).save(mask_dir / f"{Path(key).stem}.png")
            per_image.append(
                {
                    "key": key,
                    "objects": len(polygons),
                    "foreground_pixels": int((mask > 0).sum()),
                    "width": int(width),
                    "height": int(height),
                }
            )
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f"IR inference {len(per_image)}/{len(ir_keys)}", flush=True)
    path = output / "submission.json"
    path.write_text(json.dumps(submission, ensure_ascii=False, allow_nan=False) + "\n")
    checked = subprocess.run(
        [sys.executable, str(kit / "make_submission.py"), "--check", str(path)],
        capture_output=True,
        text=True,
    )
    (output / "format_check.txt").write_text(checked.stdout + checked.stderr)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "settings": settings,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch_raw": checkpoint["epoch"],
        "checkpoint_metric": checkpoint["metric"],
        "backend": backend,
        "training_config_sha256": file_sha256(Path(settings["training_config"])),
        "test_list_sha256": file_sha256(kit / "test_images.csv"),
        "submission_sha256": file_sha256(path),
        "format_check_exit_code": checked.returncode,
        "ir_images": len(ir_keys),
        "eo_images": len(keys) - len(ir_keys),
        "images_with_ir_predictions": sum(row["objects"] > 0 for row in per_image),
        "hotspot_objects": sum(row["objects"] for row in per_image),
        "reason_policy": "null",
        "eo_policy": "empty_array",
        "normal_panel_policy": "not_predicted",
        "uploaded": False,
        "external_pretraining": task.get("external_pretraining"),
        "images": per_image,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "inference_config.yaml").write_text(Path(args.config).read_text())
    print(checked.stdout, flush=True)
    if checked.returncode:
        raise RuntimeError("Official submission format check failed; inspect format_check.txt")
    print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    main()
