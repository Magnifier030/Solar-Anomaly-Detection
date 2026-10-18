"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

from research_training.task import ResearchTask


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML/JSON and validate the engine-level contract."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"name", "task", "training"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Missing required configuration keys: {sorted(missing)}")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_queue(path: Path) -> list[dict[str, Any]]:
    """Load either one experiment or a defaults-plus-experiments queue."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    if "experiments" not in value:
        return [load_config(path)]
    defaults = value.get("defaults", {})
    experiments = value["experiments"]
    if not isinstance(defaults, dict) or not isinstance(experiments, list) or not experiments:
        raise ValueError("Queue needs mapping defaults and a non-empty experiments list")
    resolved: list[dict[str, Any]] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each queued experiment must be a mapping")
        config = _merge(defaults, experiment)
        missing = {"name", "task", "training"} - config.keys()
        if missing:
            raise ValueError(f"Queued experiment misses keys: {sorted(missing)}")
        resolved.append(config)
    names = [str(config["name"]) for config in resolved]
    if len(set(names)) != len(names):
        raise ValueError("Experiment names must be unique within a queue")
    return resolved


def config_hash(config: dict[str, Any]) -> str:
    """Hash behavior-affecting configuration while allowing tracking mode changes."""
    stable = json.loads(json.dumps(config))
    stable.pop("tracking", None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def load_task(factory_path: str) -> ResearchTask:
    """Instantiate a `package.module:factory` task plugin."""
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("task.factory must use package.module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return cast(ResearchTask, factory())
