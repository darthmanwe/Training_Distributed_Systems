"""Tests for checkpoint manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.infra.config import TrainerConfig
from src.infra.seeding import capture_rng_state
from src.trainer.checkpoint import CheckpointManager
from src.trainer.ppo import PPOTrainer, compute_gae
from tests.test_ppo import _make_trajectory

if TYPE_CHECKING:
    from pathlib import Path


class TestCheckpointManager:
    def test_save_and_load(self, tmp_output_dir: Path) -> None:
        mgr = CheckpointManager(tmp_output_dir, keep=3)
        cfg = TrainerConfig(normalize_obs=False, normalize_reward=False, total_timesteps=1000)
        trainer = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))

        traj = _make_trajectory(length=32, seed=42)
        batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
        trainer.update(batch)

        state = trainer.get_checkpoint_state()
        path = mgr.save(step=1, trainer_state=state, config_dict={"seed": 42})
        assert path.exists()

        loaded = mgr.load(path)
        assert loaded["global_step"] == 1
        assert "model" in loaded
        assert "optimizer" in loaded

    def test_find_latest(self, tmp_output_dir: Path) -> None:
        mgr = CheckpointManager(tmp_output_dir, keep=5)
        state = {"model": {}, "optimizer": {}, "scheduler": {}, "global_step": 0, "rng": capture_rng_state()}

        mgr.save(step=1, trainer_state=state, config_dict={})
        mgr.save(step=2, trainer_state={**state, "global_step": 2}, config_dict={})

        latest = mgr.find_latest()
        assert latest is not None
        loaded = mgr.load(latest)
        assert loaded["global_step"] == 2

    def test_retention(self, tmp_output_dir: Path) -> None:
        mgr = CheckpointManager(tmp_output_dir, keep=2)
        state = {"model": {}, "optimizer": {}, "scheduler": {}, "global_step": 0, "rng": capture_rng_state()}

        for i in range(5):
            mgr.save(step=i, trainer_state={**state, "global_step": i}, config_dict={})

        ckpt_dir = tmp_output_dir / "checkpoints"
        step_files = list(ckpt_dir.glob("step_*.pt"))
        # latest.pt + 2 retained step files
        assert len(step_files) <= 3  # keep=2 step files + possible latest.pt

    def test_atomic_write(self, tmp_output_dir: Path) -> None:
        mgr = CheckpointManager(tmp_output_dir, keep=3)
        state = {"model": {}, "optimizer": {}, "scheduler": {}, "global_step": 5, "rng": capture_rng_state()}
        path = mgr.save(step=5, trainer_state=state, config_dict={})

        # No .tmp files should remain
        tmp_files = list((tmp_output_dir / "checkpoints").glob("*.tmp"))
        assert len(tmp_files) == 0
        assert path.exists()

    def test_verify(self, tmp_output_dir: Path) -> None:
        mgr = CheckpointManager(tmp_output_dir, keep=3)
        state = {"model": {"w": torch.tensor([1.0])}, "optimizer": {}, "scheduler": {}, "global_step": 1, "rng": capture_rng_state()}
        path = mgr.save(step=1, trainer_state=state, config_dict={})
        assert mgr.verify(path)
