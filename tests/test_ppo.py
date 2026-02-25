"""Tests for PPO algorithm: GAE computation, single step, determinism."""

from __future__ import annotations

import numpy as np
import torch

from src.infra.config import TrainerConfig
from src.trainer.normalization import ObservationNormalizer, RunningMeanStd
from src.trainer.ppo import PPOTrainer, compute_gae
from src.workers.trajectory import Trajectory


def _make_trajectory(
    length: int = 5,
    truncated: bool = False,
    last_value: float = 0.0,
    seed: int = 0,
) -> Trajectory:
    torch.manual_seed(seed)
    return Trajectory(
        observations=torch.randn(length, 4),
        actions=torch.randint(0, 2, (length,)),
        rewards=torch.randn(length),
        dones=torch.zeros(length),
        log_probs=torch.randn(length),
        values=torch.randn(length),
        truncated=truncated,
        last_value=last_value,
        policy_version=0,
    )


class TestComputeGAE:
    def test_shape_matches_input(self) -> None:
        traj = _make_trajectory(length=10)
        batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
        assert batch.advantages.shape == (10,)
        assert batch.returns.shape == (10,)

    def test_returns_equal_advantages_plus_values(self) -> None:
        traj = _make_trajectory(length=8)
        batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
        torch.testing.assert_close(
            batch.returns, batch.advantages + batch.values, atol=1e-5, rtol=1e-5
        )

    def test_truncation_bootstrap(self) -> None:
        """Truncated trajectory should use last_value as bootstrap."""
        traj_trunc = _make_trajectory(length=5, truncated=True, last_value=1.0, seed=0)
        traj_term = _make_trajectory(length=5, truncated=False, last_value=0.0, seed=0)

        batch_trunc = compute_gae([traj_trunc], gamma=0.99, gae_lambda=0.95)
        batch_term = compute_gae([traj_term], gamma=0.99, gae_lambda=0.95)

        # Advantages should differ because of different bootstrap values
        assert not torch.allclose(batch_trunc.advantages, batch_term.advantages)

    def test_known_values(self) -> None:
        """Hand-computed GAE for a simple 3-step trajectory."""
        traj = Trajectory(
            observations=torch.zeros(3, 4),
            actions=torch.zeros(3, dtype=torch.long),
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            dones=torch.tensor([0.0, 0.0, 1.0]),  # terminal at step 3
            log_probs=torch.zeros(3),
            values=torch.tensor([0.5, 1.0, 1.5]),
            truncated=False,
            last_value=0.0,
        )
        gamma, lam = 0.99, 0.95
        batch = compute_gae([traj], gamma=gamma, gae_lambda=lam)

        # Step 2 (terminal): delta = 3.0 + 0.0 - 1.5 = 1.5; adv = 1.5
        # Step 1: delta = 2.0 + 0.99*1.5 - 1.0 = 2.485; adv = 2.485 + 0.99*0.95*1.5 = 2.485 + 1.41075 = 3.89575
        # Step 0: delta = 1.0 + 0.99*1.0 - 0.5 = 1.49; adv = 1.49 + 0.99*0.95*3.89575
        expected_adv2 = 1.5
        expected_adv1 = 2.485 + gamma * lam * expected_adv2
        expected_adv0 = 1.49 + gamma * lam * expected_adv1

        torch.testing.assert_close(
            batch.advantages,
            torch.tensor([expected_adv0, expected_adv1, expected_adv2]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_multiple_trajectories(self) -> None:
        t1 = _make_trajectory(length=5, seed=0)
        t2 = _make_trajectory(length=3, seed=1)
        batch = compute_gae([t1, t2], gamma=0.99, gae_lambda=0.95)
        assert batch.size == 8


class TestPPOTrainer:
    def _make_config(self) -> TrainerConfig:
        return TrainerConfig(
            lr=3e-4,
            batch_size=32,
            num_epochs=2,
            num_minibatches=2,
            total_timesteps=1000,
            normalize_obs=False,
            normalize_reward=False,
        )

    def test_single_update(self) -> None:
        cfg = self._make_config()
        trainer = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))
        traj = _make_trajectory(length=32, seed=42)
        batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
        metrics = trainer.update(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert trainer.global_step == 1

    def test_deterministic(self) -> None:
        """Two runs with same seed produce same metrics."""
        cfg = self._make_config()

        torch.manual_seed(42)
        t1 = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))
        traj1 = _make_trajectory(length=32, seed=99)
        batch1 = compute_gae([traj1], gamma=0.99, gae_lambda=0.95)
        m1 = t1.update(batch1)

        torch.manual_seed(42)
        t2 = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))
        traj2 = _make_trajectory(length=32, seed=99)
        batch2 = compute_gae([traj2], gamma=0.99, gae_lambda=0.95)
        m2 = t2.update(batch2)

        assert abs(m1["policy_loss"] - m2["policy_loss"]) < 1e-5

    def test_checkpoint_roundtrip(self) -> None:
        cfg = self._make_config()
        trainer = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))
        traj = _make_trajectory(length=32, seed=42)
        batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
        trainer.update(batch)

        state = trainer.get_checkpoint_state()

        trainer2 = PPOTrainer(obs_dim=4, act_dim=2, config=cfg, device=torch.device("cpu"))
        trainer2.load_checkpoint_state(state)

        assert trainer2.global_step == trainer.global_step
        for k in trainer.get_policy_state_dict():
            torch.testing.assert_close(
                trainer.get_policy_state_dict()[k],
                trainer2.get_policy_state_dict()[k],
            )


class TestNormalization:
    def test_running_mean_std(self) -> None:
        rms = RunningMeanStd(shape=(3,))
        data = np.random.randn(100, 3)
        rms.update(data)
        np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=0.1)

    def test_running_mean_std_state_dict(self) -> None:
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        state = rms.state_dict()

        rms2 = RunningMeanStd(shape=(2,))
        rms2.load_state_dict(state)
        np.testing.assert_array_equal(rms.mean, rms2.mean)

    def test_observation_normalizer(self) -> None:
        norm = ObservationNormalizer(obs_dim=4)
        obs = torch.randn(10, 4)
        out = norm.normalize(obs)
        assert out.shape == (10, 4)
        # After normalization, values should be roughly centered
        assert out.abs().max() <= 10.0
