"""Optional W&B experiment tracking with a discoverable browser URL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class WandbTracker:
    """Log bounded summaries while JSONL remains the local source of truth."""

    def __init__(self, config: dict[str, Any], directory: Path) -> None:
        settings = config.get("tracking", {}).get("wandb", {})
        mode = str(settings.get("mode", "disabled"))
        if mode not in {"disabled", "offline", "online"}:
            raise ValueError("tracking.wandb.mode must be disabled, offline or online")
        self.run: Any = None
        self.project = str(settings.get("project", "research-training"))
        self.entity = settings.get("entity")
        self.run_id = hashlib.sha256(str(config["name"]).encode()).hexdigest()[:16]
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install the [wandb] extra to enable W&B tracking") from error
        options: dict[str, Any] = {
            "project": self.project,
            "name": str(config["name"]),
            "id": self.run_id,
            "resume": "allow",
            "mode": mode,
            "dir": str(directory),
            "config": config,
            "tags": list(settings.get("tags", ["research-training"])),
            "settings": wandb.Settings(
                x_disable_stats=not bool(settings.get("system_monitoring", True))
            ),
        }
        if self.entity:
            options["entity"] = str(self.entity)
        self.run = wandb.init(**options)
        if self.run is None:
            raise RuntimeError("W&B did not create a run")
        self.run.define_metric("optimizer_step")
        self.run.define_metric("train/*", step_metric="optimizer_step")
        self.run.define_metric("validation/*", step_metric="optimizer_step")
        run_url = getattr(self.run, "url", None)
        (directory / "wandb_run.json").write_text(
            json.dumps({"run_id": self.run_id, "project": self.project, "url": run_url}, indent=2),
            encoding="utf-8",
        )

    def log(self, metrics: dict[str, Any]) -> None:
        """Send one summary to W&B."""
        if self.run is not None:
            self.run.log({key: value for key, value in metrics.items() if value is not None})

    def summary(self, metrics: dict[str, Any]) -> None:
        """Store final scalar metrics in the W&B run summary."""
        if self.run is not None:
            self.run.summary.update(metrics)

    def artifact(self, path: Path, name: str | None = None) -> None:
        """Upload a final run file as a W&B Artifact when tracking is enabled."""
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name or path.stem, type="model" if path.suffix == ".pt" else "run-file")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Flush and close W&B."""
        if self.run is not None:
            self.run.finish()
