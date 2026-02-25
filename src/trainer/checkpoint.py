"""Checkpoint manager: atomic save/load of full training state."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from src.infra.errors import CheckpointCorruptionError

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"model", "optimizer", "scheduler", "global_step", "rng", "config"}


class CheckpointManager:
    """Saves and loads training checkpoints with atomic writes and retention policy."""

    def __init__(self, run_dir: Path, keep: int = 3):
        self._ckpt_dir = run_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._keep = keep

    def save(self, step: int, trainer_state: dict[str, Any], config_dict: dict) -> Path:
        """Atomic save: write to .tmp then rename."""
        state = {**trainer_state, "config": config_dict}

        filename = f"step_{step:06d}.pt"
        target = self._ckpt_dir / filename
        tmp = target.with_suffix(".tmp")

        torch.save(state, str(tmp))
        # os.replace is atomic on both Windows NTFS and POSIX
        os.replace(str(tmp), str(target))

        self._update_latest_link(target)
        self._enforce_retention()

        logger.info("checkpoint.saved", extra={"step": step, "path": str(target)})
        return target

    def load(self, path: Path | str) -> dict[str, Any]:
        """Load checkpoint with integrity validation."""
        path = Path(path)
        if not path.exists():
            raise CheckpointCorruptionError(str(path), "file not found")

        state = torch.load(str(path), map_location="cpu", weights_only=False)

        missing = REQUIRED_KEYS - set(state.keys())
        if missing:
            raise CheckpointCorruptionError(str(path), f"missing keys: {missing}")

        logger.info("checkpoint.loaded", extra={"path": str(path), "step": state["global_step"]})
        result: dict[str, Any] = state
        return result

    def find_latest(self) -> Path | None:
        """Find the most recent checkpoint by step number."""
        latest = self._ckpt_dir / "latest.pt"
        if latest.exists():
            return latest

        checkpoints = sorted(self._ckpt_dir.glob("step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def verify(self, path: Path) -> bool:
        """Load and verify a checkpoint has all expected keys."""
        try:
            state = self.load(path)
            return "model" in state and "optimizer" in state
        except Exception:
            return False

    def _update_latest_link(self, target: Path) -> None:
        """Create/update latest.pt as a copy (Windows doesn't support symlinks reliably)."""
        latest = self._ckpt_dir / "latest.pt"
        tmp = latest.with_suffix(".tmp")
        torch.save(torch.load(str(target), map_location="cpu", weights_only=False), str(tmp))
        os.replace(str(tmp), str(latest))

    def _enforce_retention(self) -> None:
        """Keep only the last K checkpoints plus latest.pt."""
        checkpoints = sorted(self._ckpt_dir.glob("step_*.pt"))
        while len(checkpoints) > self._keep:
            old = checkpoints.pop(0)
            old.unlink(missing_ok=True)
            logger.debug("checkpoint.deleted", extra={"path": str(old)})
