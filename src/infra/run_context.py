"""Run context: output directory creation, metadata capture, config persistence."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from src.infra.config import AppConfig


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def setup_run_directory(cfg: AppConfig, base_dir: str = "outputs") -> Path:
    """Create the output directory tree and write config + metadata files."""
    run_dir = Path(base_dir) / str(cfg.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "profiling").mkdir(exist_ok=True)

    cfg_dict = cfg.model_dump()
    omega = OmegaConf.create(cfg_dict)
    with open(run_dir / "config_resolved.yaml", "w") as f:
        OmegaConf.save(omega, f)

    metadata = {
        "run_id": cfg.run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": _get_git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": cfg.seed,
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir
