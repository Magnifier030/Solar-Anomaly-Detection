"""Task-agnostic PyTorch training, evaluation, tracking, and exact state resume."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from research_training.config import config_hash, load_task
from research_training.storage import ArtifactStore, atomic_replace, file_sha256
from research_training.task import TaskBundle
from research_training.tracking import WandbTracker


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _gpu_utilization(device: torch.device) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "system/gpu_utilization_pct": None,
        "system/gpu_board_memory_used_mb": None,
        "system/gpu_board_memory_total_mb": None,
    }
    if device.type != "cuda":
        return result
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        fields = process.stdout.strip().splitlines()[0].split(",")
        result["system/gpu_utilization_pct"] = float(fields[0].strip())
        result["system/gpu_board_memory_used_mb"] = float(fields[1].strip())
        result["system/gpu_board_memory_total_mb"] = float(fields[2].strip())
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        IndexError,
        ValueError,
    ):
        pass
    return result


def _save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_replace(temporary, path)


def _write_logs(directory: Path, history: list[dict[str, Any]]) -> None:
    _save_json(directory / "history.json", {"events": history})
    temporary = directory / "train_log.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for event in history:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    atomic_replace(temporary, directory / "train_log.jsonl")


def _prepare_artifacts(config: dict[str, Any], store: ArtifactStore) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    artifacts = config.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")
    for item in artifacts:
        if not isinstance(item, dict) or "name" not in item or "path" not in item:
            raise ValueError("Each artifact needs name and path")
        path = Path(str(item["path"]))
        local = store.ensure_local(path, item.get("drive_file_id"), item.get("sha256"))
        record = {
            "name": str(item["name"]),
            "path": str(local),
            "sha256": file_sha256(local),
            "version": item.get("version"),
            "drive_file_id": item.get("drive_file_id"),
        }
        item["resolved_sha256"] = record["sha256"]
        manifest.append(record)
    return manifest


def _checkpoint(
    *,
    config: dict[str, Any],
    bundle: TaskBundle,
    scaler: Any,
    epoch: int,
    next_batch: int,
    step: int,
    best_metric: float,
    best_epoch: int,
    patience_count: int,
    best_weights: dict[str, torch.Tensor] | None,
    history: list[dict[str, Any]],
    completed: bool,
    epoch_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config": config,
        "config_hash": config_hash(config),
        "git_revision": _git_revision(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "metadata": bundle.metadata,
        "model": bundle.model.state_dict(),
        "optimizer": bundle.optimizer.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "best_weights": best_weights,
        "history": history,
        "completed": completed,
        "rng": _rng_state(),
        "epoch_generator_state": epoch_generator_state,
    }


def run_experiment(
    config: dict[str, Any],
    store: ArtifactStore,
    *,
    seed: int,
    device_name: str = "auto",
    resume: bool = True,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train one configured task with resumable state and remote weight publication."""
    name = str(config["name"])
    directory = store.run_dir(name)
    resolved = copy.deepcopy(config)
    resolved.setdefault("training", {})["seed"] = seed
    manifest = _prepare_artifacts(resolved, store)
    _save_json(directory / "resolved_config.json", resolved)
    _save_json(directory / "data_manifest.json", {"artifacts": manifest})
    store.upload_artifact(name, directory / "resolved_config.json")
    store.upload_artifact(name, directory / "data_manifest.json")
    tracker = WandbTracker(resolved, directory)
    try:
        return _run(resolved, store, tracker, seed, device_name, resume, max_steps)
    finally:
        tracker.finish()


def _run(
    config: dict[str, Any],
    store: ArtifactStore,
    tracker: WandbTracker,
    seed: int,
    device_name: str,
    resume: bool,
    max_steps: int | None,
) -> dict[str, Any]:
    training = config["training"]
    name = str(config["name"])
    directory = store.run_dir(name)
    device = _device(device_name)
    _seed_everything(seed)
    task = load_task(str(config["task"]["factory"]))
    bundle = task.build(config, device)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler: Any = torch.amp.GradScaler("cuda", enabled=amp)
    selection = str(training.get("selection_metric", "loss"))
    direction = str(training.get("selection_mode", "min"))
    if direction not in {"min", "max"}:
        raise ValueError("training.selection_mode must be min or max")
    best_metric = float("inf") if direction == "min" else -float("inf")
    best_epoch, patience_count, step, start_epoch, next_batch = -1, 0, 0, 0, 0
    best_weights: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    epoch_generator_state: torch.Tensor | None = None
    checkpoint = store.restore_checkpoint(name) if resume else None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["config_hash"] != config_hash(config):
            raise ValueError("Refusing to resume with changed behavior-affecting configuration")
        if state["completed"]:
            if "result" in state:
                return cast(dict[str, Any], state["result"])
            return cast(
                dict[str, Any],
                json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
            )
        bundle.model.load_state_dict(state["model"])
        bundle.optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["grad_scaler"])
        start_epoch, next_batch, step = state["epoch"], state["next_batch"], state["global_step"]
        best_metric, best_epoch = state["best_metric"], state["best_epoch"]
        patience_count, best_weights, history = (
            state["patience_count"], state["best_weights"], state["history"]
        )
        epoch_generator_state = state.get("epoch_generator_state")
        _restore_rng(state["rng"])
    maximum_epochs = int(training["max_epochs"])
    checkpoint_every = int(training.get("checkpoint_every_steps", 100))
    patience = int(training.get("patience", maximum_epochs))
    gradient_clip = training.get("gradient_clip")
    stopped_early = False
    training_started = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs):
        bundle.model.train()
        if bundle.train_generator is not None:
            if epoch == start_epoch and next_batch and epoch_generator_state is not None:
                bundle.train_generator.set_state(epoch_generator_state)
            epoch_generator_state = bundle.train_generator.get_state()
        iterator = iter(bundle.train_loader)
        for _ in range(next_batch):
            next(iterator)
        for batch_index, batch in enumerate(iterator, start=next_batch):
            started = time.perf_counter()
            bundle.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, diagnostics = task.training_loss(bundle.model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(bundle.optimizer)
            gradient_norm = None
            if gradient_clip is not None:
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    bundle.model.parameters(), float(gradient_clip)
                ))
            scaler.step(bundle.optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step += 1
            duration = time.perf_counter() - started
            try:
                batch_size = len(batch[0])
            except (TypeError, IndexError):
                batch_size = 1
            event = {
                "kind": "iteration", "epoch": epoch, "batch": batch_index,
                "optimizer_step": step, "train/loss": float(loss.detach()),
                "train/gradient_norm": gradient_norm,
                "train/iteration_seconds": duration,
                "train/throughput_samples_per_s": batch_size / max(duration, 1e-9),
                "train/learning_rate": bundle.optimizer.param_groups[0]["lr"],
                "system/gpu_memory_allocated_mb": (
                    torch.cuda.memory_allocated(device) / (1024**2)
                    if device.type == "cuda" else None
                ),
                "system/gpu_peak_memory_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda" else None
                ),
                **{f"train/{key}": value for key, value in diagnostics.items()},
            }
            history.append(event)
            if step % checkpoint_every == 0 or (max_steps is not None and step >= max_steps):
                payload = _checkpoint(
                    config=config, bundle=bundle, scaler=scaler, epoch=epoch,
                    next_batch=batch_index + 1, step=step, best_metric=best_metric,
                    best_epoch=best_epoch, patience_count=patience_count,
                    best_weights=best_weights, history=history, completed=False,
                    epoch_generator_state=epoch_generator_state,
                )
                store.save_checkpoint(name, payload, step)
                _write_logs(directory, history)
                for artifact in (directory / "train_log.jsonl", directory / "history.json"):
                    store.upload_artifact(name, artifact)
                tracker.log(event)
            if max_steps is not None and step >= max_steps:
                return {"status": "paused_at_step_limit", "global_step": step}
        next_batch = 0
        validation = task.validation_metrics(bundle.model, bundle.validation_loader, device)
        if selection not in validation:
            raise KeyError(f"Selection metric {selection!r} missing from validation metrics")
        current = float(validation[selection])
        improved = current < best_metric if direction == "min" else current > best_metric
        if improved:
            best_metric, best_epoch, patience_count = current, epoch, 0
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in bundle.model.state_dict().items()
            }
            temporary = directory / "best.pt.tmp"
            torch.save(
                {"model": best_weights, "config": config, "metric": current, "epoch": epoch},
                temporary,
            )
            atomic_replace(temporary, directory / "best.pt")
        else:
            patience_count += 1
        event = {"kind": "epoch", "epoch": epoch, "optimizer_step": step, **{
            f"validation/{key}": value for key, value in validation.items()
        }, **_gpu_utilization(device)}
        elapsed = time.perf_counter() - training_started
        completed_epochs = epoch - start_epoch + 1
        event["train/eta_to_max_epochs_s"] = (
            elapsed / completed_epochs * max(0, maximum_epochs - epoch - 1)
        )
        history.append(event)
        tracker.log(event)
        payload = _checkpoint(
            config=config, bundle=bundle, scaler=scaler, epoch=epoch + 1, next_batch=0,
            step=step, best_metric=best_metric, best_epoch=best_epoch,
            patience_count=patience_count, best_weights=best_weights, history=history,
            completed=False, epoch_generator_state=None,
        )
        store.save_checkpoint(name, payload, step)
        _write_logs(directory, history)
        for artifact in (directory / "train_log.jsonl", directory / "history.json"):
            store.upload_artifact(name, artifact)
        if (directory / "best.pt").exists():
            store.upload_artifact(name, directory / "best.pt")
        if patience_count >= patience:
            stopped_early = True
            break
    if best_weights is None:
        raise RuntimeError("Training completed without a selectable checkpoint")
    bundle.model.load_state_dict(best_weights)
    final_validation = task.validation_metrics(bundle.model, bundle.validation_loader, device)
    metrics: dict[str, Any] = {
        "status": "complete", "seed": seed, "global_step": step,
        "best_epoch": best_epoch, "best_metric": best_metric,
        "stopped_early": stopped_early, "validation": final_validation,
        "metadata": bundle.metadata,
    }
    _save_json(directory / "metrics.json", metrics)
    tracker.summary({f"result/{key}": value for key, value in final_validation.items()})
    payload = _checkpoint(
        config=config, bundle=bundle, scaler=scaler, epoch=best_epoch,
        next_batch=0, step=step, best_metric=best_metric, best_epoch=best_epoch,
        patience_count=patience_count, best_weights=best_weights, history=history,
        completed=True, epoch_generator_state=None,
    )
    payload["result"] = metrics
    store.save_checkpoint(name, payload, step)
    final_artifacts = (
        directory / "best.pt",
        directory / "metrics.json",
        directory / "resolved_config.json",
    )
    for artifact in final_artifacts:
        store.upload_artifact(name, artifact)
        tracker.artifact(artifact, f"{name}-{artifact.stem}")
    artifact_hook = getattr(task, "artifacts", None)
    if callable(artifact_hook):
        for artifact in artifact_hook(config, directory):
            path = Path(artifact)
            if not path.is_file() or directory not in path.resolve().parents:
                raise ValueError(f"Task artifact must be a file inside the run directory: {path}")
            store.upload_artifact(name, path)
    return metrics
