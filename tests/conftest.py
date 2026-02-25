"""Shared test fixtures for the distributed RL platform."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def tmp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(prefix="distrl_test_") as d:
        yield Path(d)


@pytest.fixture
def base_config_dict() -> dict:
    """Minimal valid config dict for testing."""
    return {
        "seed": 42,
        "run_id": "test_run",
        "env": {
            "name": "cartpole",
            "num_env_workers": 3,
            "num_jobs": 10,
            "max_queue_len": 5,
            "max_steps_per_episode": 100,
        },
        "coordinator": {
            "heartbeat_timeout_s": 10.0,
            "heartbeat_interval_s": 2.0,
            "persist_state": False,
        },
        "workers": {
            "num_workers": 2,
            "rollout_steps": 64,
            "default_speed_factor": 1.0,
            "default_failure_rate": 0.0,
            "default_max_batch": 256,
            "heterogeneous": False,
            "profiles": [
                {"speed_factor": 1.0, "failure_rate": 0.0, "max_batch": 256},
                {"speed_factor": 1.0, "failure_rate": 0.0, "max_batch": 256},
            ],
        },
        "trainer": {
            "algorithm": "ppo",
            "total_timesteps": 1000,
            "batch_size": 128,
            "num_epochs": 2,
            "num_minibatches": 2,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "normalize_obs": False,
            "normalize_reward": False,
            "eval_interval": 5,
            "eval_episodes": 2,
            "checkpoint_interval": 5,
            "checkpoint_keep": 2,
        },
        "churn": {
            "enabled": False,
            "kill_interval_s": 60,
            "kill_probability": 0.0,
        },
        "perf": {
            "compile": False,
            "mixed_precision": False,
            "centralized_inference": False,
            "profile": False,
            "profile_steps": 5,
        },
        "obs": {
            "tracker": "local",
            "prometheus_port": 8000,
            "log_level": "INFO",
        },
    }
