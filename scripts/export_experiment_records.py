"""Export auditable research results without modifying experiment outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="docs/experiments/results_20261022")
    args = parser.parse_args()
    records = []
    workspace = Path.cwd().resolve()
    active = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = proc.joinpath("cmdline").read_bytes().decode().split("\0")
            if (
                (
                    "research_training.cli" in cmd
                    or any(Path(x).name == "research-train" for x in cmd)
                )
                and "run" in cmd
                and proc.joinpath("cwd").resolve() == workspace
            ):
                active.append({"pid": int(proc.name), "command": cmd})
        except (OSError, UnicodeError):
            continue
    for run in sorted(Path(args.outputs).iterdir()):
        resolved_path = run / "resolved_config.json"
        if not resolved_path.is_file():
            continue
        resolved = json.loads(resolved_path.read_text())
        saved, checkpoint = resolved, {}
        latest = run / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            saved = checkpoint["config"]
        history_path = run / "history.json"
        events = (
            json.loads(history_path.read_text()).get("events", []) if history_path.exists() else []
        )
        epochs = [event for event in events if event.get("kind") == "epoch"]
        metrics_path = run / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        training, task = saved.get("training", {}), saved.get("task", {})
        selection = training.get("selection_metric")
        choose = min if training.get("selection_mode") == "min" else max
        best = (
            choose(epochs, key=lambda row: row.get(f"validation/{selection}", 0)) if epochs else {}
        )
        best_epoch = metrics.get("best_epoch", best.get("epoch"))
        metadata = metrics.get("metadata", checkpoint.get("metadata", {}))
        completed = metrics.get("status") == "complete" or checkpoint.get("completed", False)
        differences = {
            key: {"checkpoint": saved.get(key), "resolved": resolved.get(key)}
            for key in ["task", "training", "runtime"]
            if saved.get(key) != resolved.get(key)
        }
        record = {
            "run": run.name,
            "path": str(run),
            "status": "complete"
            if completed
            else "incomplete_saved_run"
            if latest.exists()
            else "config_only_no_training_evidence",
            "epochs_recorded": len(epochs),
            "last_completed_epoch_display": epochs[-1]["epoch"] + 1 if epochs else None,
            "best_epoch_raw": best_epoch,
            "best_epoch_display": best_epoch + 1 if best_epoch is not None else None,
            "selection_metric": selection,
            "best_metric": metrics.get("best_metric", best.get(f"validation/{selection}")),
            "best_validation": metrics.get(
                "validation",
                {
                    key.removeprefix("validation/"): value
                    for key, value in best.items()
                    if key.startswith("validation/")
                },
            ),
            "last_validation": {
                key.removeprefix("validation/"): value
                for key, value in (epochs[-1] if epochs else {}).items()
                if key.startswith("validation/")
            },
            "saved_global_step": checkpoint.get("global_step", metrics.get("global_step")),
            "checkpoint_epoch_raw": checkpoint.get("epoch"),
            "checkpoint_next_batch": checkpoint.get("next_batch"),
            "model": task.get("model", {}),
            "image_size": task.get("image_size"),
            "split_strategy": task.get("split_strategy"),
            "validation_groups": task.get("validation_groups"),
            "preprocessing": task.get("preprocessing"),
            "training": training,
            "train_samples": metadata.get("train_samples"),
            "validation_samples": metadata.get("validation_samples"),
            "finetuning": metadata.get("finetuning"),
            "metric_aggregation": metadata.get(
                "metric_aggregation", task.get("metric_aggregation")
            ),
            "config_source": str(latest) if latest.exists() else str(resolved_path),
            "resolved_config_sha256": digest(resolved_path),
            "resolved_checkpoint_differences": differences,
            "metrics_sha256": digest(metrics_path) if metrics_path.exists() else None,
            "best_checkpoint_sha256": digest(run / "best.pt")
            if (run / "best.pt").exists()
            else None,
            "artifact_paths": [str(p) for p in run.iterdir() if p.is_file()],
        }
        records.append(record)
        del checkpoint, events
    result = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "epoch_convention": "display=raw+1; epochs_recorded counts completed validation events",
        "active_project_training_processes": active,
        "runs": records,
        "notes": [
            "No historical missing defaults are inferred from current code.",
            "Incomplete runs have no claim of full training completion.",
            "Different splits, resolutions and aggregation policies are not matched comparisons.",
        ],
    }
    evaluation_sources = {
        "transfer_validation": Path(args.outputs)
        / "solar-ir-dnanet-1ch-pvhsd-finetune-cudnn/analysis"
        / "transfer_validation_20261022/result.json",
        "test_analysis": Path(args.outputs)
        / "submissions/dnanet-ir-only-photo-20261022/analysis/summary.json",
        "platform_result_user_reported": Path(args.outputs)
        / "submissions/dnanet-ir-only-photo-20261022/analysis/platform_result.json",
        "drive_backup": Path(args.outputs)
        / "submissions/dnanet-ir-only-photo-20261022/drive_backup_receipt.json",
        "pretraining_stop": Path(args.outputs)
        / "pvhsd-dnanet-1ch-pretrain/analysis/pretraining_stop.json",
        "cpu_verification_only": Path(args.outputs) / "verification/pvhsd2025_1ch/result.json",
        "cudnn_verification_only": Path(args.outputs) / "verification/cudnn_backend/result.json",
        "frozen_state_verification": Path(args.outputs)
        / "verification/finetuning_policy/freeze_runtime_audit.json",
    }
    result["evaluations"] = {}
    for label, path in evaluation_sources.items():
        if not path.exists():
            continue
        content = json.loads(path.read_text())
        if label == "transfer_validation":
            content["models"] = {
                name: {key: value for key, value in model.items() if key != "images"}
                for name, model in content["models"].items()
            }
        result["evaluations"][label] = {
            "source": str(path),
            "sha256": digest(path),
            "result": content,
        }
    prefix = Path(args.destination)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "run",
        "status",
        "epochs_recorded",
        "best_epoch_raw",
        "best_epoch_display",
        "selection_metric",
        "best_metric",
        "saved_global_step",
        "image_size",
        "split_strategy",
        "train_samples",
        "validation_samples",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    print(f"Recorded {len(records)} runs; {len(active)} active project training processes")


if __name__ == "__main__":
    main()
