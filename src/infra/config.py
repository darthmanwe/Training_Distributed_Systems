"""Configuration system: OmegaConf for YAML loading/composition, Pydantic for validation.

Usage:
    cfg = load_config("configs/base.yaml")
    cfg = load_config("configs/base.yaml", overrides=["trainer.lr=1e-3", "seed=123"])
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class EnvConfig(BaseModel):
    name: str = "scheduling"
    num_env_workers: int = 5
    num_jobs: int = 20
    max_queue_len: int = 10
    max_steps_per_episode: int = 200


class CoordinatorConfig(BaseModel):
    heartbeat_timeout_s: float = 15.0
    heartbeat_interval_s: float = 5.0
    persist_state: bool = True


class WorkerProfile(BaseModel):
    speed_factor: float = 1.0
    failure_rate: float = 0.0
    max_batch: int = 512


class WorkersConfig(BaseModel):
    num_workers: int = 4
    rollout_steps: int = 256
    default_speed_factor: float = 1.0
    default_failure_rate: float = 0.0
    default_max_batch: int = 512
    heterogeneous: bool = True
    profiles: list[WorkerProfile] = Field(default_factory=list)


class TrainerConfig(BaseModel):
    algorithm: str = "ppo"
    total_timesteps: int = 100_000
    batch_size: int = 1024
    num_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    normalize_obs: bool = True
    normalize_reward: bool = True
    eval_interval: int = 10
    eval_episodes: int = 5
    checkpoint_interval: int = 10
    checkpoint_keep: int = 3


class ChurnConfig(BaseModel):
    enabled: bool = False
    kill_interval_s: float = 60.0
    kill_probability: float = 0.0


class PerfConfig(BaseModel):
    compile: bool = False
    mixed_precision: bool = False
    centralized_inference: bool = False
    profile: bool = False
    profile_steps: int = 5


class ObsConfig(BaseModel):
    tracker: str = "local"
    prometheus_port: int = 8000
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Root configuration validated by Pydantic after OmegaConf resolution."""

    seed: int = 42
    run_id: str | None = None
    env: EnvConfig = Field(default_factory=EnvConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)

    def model_post_init(self, __context: object) -> None:
        if self.run_id is None:
            self.run_id = _generate_run_id(self)
        n_profiles = len(self.workers.profiles)
        if n_profiles > 0 and n_profiles < self.workers.num_workers:
            last = self.workers.profiles[-1]
            while len(self.workers.profiles) < self.workers.num_workers:
                self.workers.profiles.append(last.model_copy())


def _generate_run_id(cfg: AppConfig) -> str:
    """Timestamp + short config hash for unique but traceable run IDs."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(cfg.seed).encode() + str(cfg.trainer.lr).encode()).hexdigest()[:6]
    return f"{ts}_{h}"


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    merge_with: str | Path | None = None,
) -> AppConfig:
    """Load and validate config from YAML, optionally merging a second file and CLI overrides."""
    base = OmegaConf.load(str(path))

    if merge_with is not None:
        overlay = OmegaConf.load(str(merge_with))
        base = OmegaConf.merge(base, overlay)

    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli)

    raw = OmegaConf.to_container(base, resolve=True)
    assert isinstance(raw, dict)
    return AppConfig(**raw)  # type: ignore[arg-type]
