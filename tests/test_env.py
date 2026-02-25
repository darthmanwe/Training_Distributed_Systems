"""Tests for environment implementations."""

from __future__ import annotations

import numpy as np

from src.workers.envs.gym_env import GymEnvWrapper
from src.workers.envs.scheduling_env import JobSchedulingEnv


class TestJobSchedulingEnv:
    def test_reset_returns_correct_shape(self) -> None:
        env = JobSchedulingEnv(num_env_workers=3, max_queue_len=5, seed=42)
        obs, info = env.reset()
        expected_dim = 5 * 3 + 3 * 3  # queue * job_features + workers * worker_features
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_types(self) -> None:
        env = JobSchedulingEnv(num_env_workers=3, seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "jobs_completed" in info

    def test_episode_terminates(self) -> None:
        env = JobSchedulingEnv(num_env_workers=3, num_jobs=5, max_steps_per_episode=50, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
        assert done

    def test_wait_action(self) -> None:
        env = JobSchedulingEnv(num_env_workers=3, seed=42)
        env.reset()
        action_wait = env.action_space.n - 1
        obs, reward, _, _, _ = env.step(action_wait)
        assert obs is not None

    def test_deterministic_with_seed(self) -> None:
        env1 = JobSchedulingEnv(seed=42)
        env2 = JobSchedulingEnv(seed=42)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestGymEnvWrapper:
    def test_cartpole_reset(self) -> None:
        env = GymEnvWrapper("CartPole-v1", seed=42)
        obs, info = env.reset()
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

    def test_cartpole_step(self) -> None:
        env = GymEnvWrapper("CartPole-v1", seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
