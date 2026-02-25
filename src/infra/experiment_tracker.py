"""Experiment tracking: local JSONL, optional W&B and MLflow backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Protocol for experiment tracking backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path, name: str | None = None) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class LocalTracker(ExperimentTracker):
    """Writes metrics to a local JSONL file."""

    def __init__(self, run_dir: Path):
        self._metrics_file = run_dir / "metrics.jsonl"
        self._run_dir = run_dir

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        entry = {"step": step, **metrics}
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_config(self, config: dict[str, Any]) -> None:
        with open(self._run_dir / "config_tracked.json", "w") as f:
            json.dump(config, f, indent=2)

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        logger.debug(f"Artifact logged: {path}")

    def finish(self) -> None:
        logger.info("LocalTracker finished")


class WandbTracker(ExperimentTracker):
    """Weights & Biases tracking (requires wandb extra)."""

    def __init__(self, project: str, run_name: str, config: dict[str, Any]):
        try:
            import wandb

            self._run = wandb.init(project=project, name=run_name, config=config)
            self._wandb = wandb
        except ImportError as exc:
            raise ImportError(
                "wandb not installed. Install with: pip install 'dist-rl-platform[wandb]'"
            ) from exc

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        self._wandb.save(str(path))

    def finish(self) -> None:
        self._wandb.finish()


def create_tracker(
    tracker_type: str, run_dir: Path, config: dict[str, Any] | None = None
) -> ExperimentTracker:
    """Factory for experiment trackers."""
    if tracker_type == "wandb":
        return WandbTracker(
            project="dist-rl-platform",
            run_name=run_dir.name,
            config=config or {},
        )
    return LocalTracker(run_dir)
