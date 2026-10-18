"""Command-line interface for training, validation, and W&B frontend access."""

from __future__ import annotations

import argparse
import json
import os
import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from research_training.config import load_queue
from research_training.engine import run_experiment
from research_training.storage import ArtifactStore, DriveApiStore, RcloneStore, RemoteStore


def _store(use_drive: bool) -> ArtifactStore:
    output_root = Path(os.getenv("TRAINING_OUTPUT_ROOT", "outputs"))
    parent = os.getenv("TRAINING_GDRIVE_PARENT_FOLDER_ID", "")
    if not use_drive or not parent:
        return ArtifactStore(output_root)
    folder = os.getenv("TRAINING_GDRIVE_FOLDER_PATH", "research-training/experiments")
    backend = os.getenv("TRAINING_GDRIVE_BACKEND", "rclone")
    if backend == "rclone":
        remote: RemoteStore = RcloneStore(
            os.getenv("TRAINING_RCLONE_REMOTE", "Google"), parent, folder
        )
    elif backend == "api":
        remote = DriveApiStore(parent, folder)
    else:
        raise ValueError("TRAINING_GDRIVE_BACKEND must be rclone or api")
    return ArtifactStore(output_root, remote)


def _override(config: dict[str, Any], args: argparse.Namespace) -> None:
    settings = config.setdefault("tracking", {}).setdefault("wandb", {})
    if getattr(args, "wandb_mode", None):
        settings["mode"] = args.wandb_mode
    if getattr(args, "wandb_project", None):
        settings["project"] = args.wandb_project
    entity = os.getenv("WANDB_ENTITY")
    if entity and not settings.get("entity"):
        settings["entity"] = entity


def _open_dashboard(run_name: str | None, project: str | None) -> None:
    if run_name:
        output_root = Path(os.getenv("TRAINING_OUTPUT_ROOT", "outputs"))
        metadata_path = output_root / run_name / "wandb_run.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No W&B metadata found at {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        url = metadata.get("url")
        if not url:
            raise RuntimeError(
                "This run was offline/disabled and has no hosted W&B URL; sync it first"
            )
    else:
        entity = os.getenv("WANDB_ENTITY")
        if not entity or not project:
            raise ValueError("Project dashboard requires WANDB_ENTITY and --project")
        url = f"https://wandb.ai/{entity}/{project}"
    print(url)
    webbrowser.open(str(url))


def _run_id(value: str | None) -> str:
    candidate = value or datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M")
    if not re.fullmatch(r"\d{8}_\d{4}", candidate):
        raise ValueError("run-id must use YYYYMMDD_HHMM")
    return candidate


def main() -> None:
    """Run the selected CLI command."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reusable research training template")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("run", help="train or resume an experiment")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--experiment", help="run only one named queue entry")
    train.add_argument("--run-id", help="YYYYMMDD_HHMM prefix; reuse to resume")
    train.add_argument("--run-name", help="optional label inserted after run-id")
    train.add_argument("--device", default="auto")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    train.add_argument("--wandb-project")
    train.add_argument("--drive", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    dashboard = subparsers.add_parser("dashboard", help="open the W&B frontend")
    dashboard.add_argument("--run-name")
    dashboard.add_argument("--project")
    args = parser.parse_args()
    if args.command == "dashboard":
        _open_dashboard(args.run_name, args.project)
        return
    configs = load_queue(args.config)
    if args.experiment:
        configs = [config for config in configs if config["name"] == args.experiment]
        if not configs:
            raise ValueError(f"Unknown queued experiment: {args.experiment}")
    prefix = _run_id(args.run_id) if (args.run_id or len(configs) > 1 or args.run_name) else None
    results: list[dict[str, Any]] = []
    store = _store(args.drive)
    for config in configs:
        _override(config, args)
        if prefix:
            components = [prefix, args.run_name, str(config["name"])]
            config["name"] = "_".join(component for component in components if component)
        seed = args.seed if args.seed is not None else int(config["training"].get("seed", 42))
        results.append(
            run_experiment(
                config,
                store,
                seed=seed,
                device_name=args.device,
                resume=args.resume,
                max_steps=args.max_steps,
            )
        )
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
