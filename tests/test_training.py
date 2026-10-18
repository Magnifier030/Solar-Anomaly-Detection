from __future__ import annotations

import copy
from pathlib import Path

import torch

from research_training.config import config_hash, load_config, load_queue
from research_training.engine import run_experiment
from research_training.storage import ArtifactStore


def _config() -> dict[str, object]:
    return {
        "name": "test-run",
        "task": {
            "factory": "research_training.examples.regression:create_task",
            "samples": 64,
            "features": 3,
            "hidden_size": 8,
        },
        "training": {
            "seed": 3,
            "batch_size": 8,
            "max_epochs": 4,
            "patience": 4,
            "learning_rate": 0.02,
            "checkpoint_every_steps": 2,
            "selection_metric": "loss",
            "selection_mode": "min",
            "amp": False,
        },
        "tracking": {"wandb": {"mode": "disabled"}},
    }


def test_smoke_run_writes_reproducibility_bundle(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    result = run_experiment(_config(), store, seed=3, device_name="cpu")
    run_dir = tmp_path / "test-run"
    assert result["status"] == "complete"
    assert result["validation"]["loss"] >= 0
    assert {"best.pt", "latest.pt", "metrics.json", "resolved_config.json", "history.json"} <= {
        path.name for path in run_dir.iterdir()
    }
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["completed"] is True
    assert checkpoint["rng"]["python"]
    assert checkpoint["config_hash"] == config_hash(checkpoint["config"])


def test_pause_and_resume_matches_uninterrupted(tmp_path: Path) -> None:
    interrupted = _config()
    uninterrupted = copy.deepcopy(interrupted)
    uninterrupted["name"] = "reference"
    store = ArtifactStore(tmp_path)
    paused = run_experiment(interrupted, store, seed=3, device_name="cpu", max_steps=3)
    assert paused == {"status": "paused_at_step_limit", "global_step": 3}
    resumed = run_experiment(interrupted, store, seed=3, device_name="cpu")
    reference = run_experiment(uninterrupted, store, seed=3, device_name="cpu")
    assert resumed["validation"] == reference["validation"]


def test_tracking_mode_does_not_change_resume_hash() -> None:
    first = _config()
    second = copy.deepcopy(first)
    second["tracking"]["wandb"]["mode"] = "online"
    assert config_hash(first) == config_hash(second)


def test_example_config_loads() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config["task"]["factory"].endswith(":create_task")


def test_queue_deep_merges_defaults() -> None:
    configs = load_queue(Path("configs/queue.example.yaml"))
    assert [config["name"] for config in configs] == ["small-model", "large-model"]
    assert configs[0]["task"]["features"] == 4
    assert configs[1]["task"]["hidden_size"] == 32
