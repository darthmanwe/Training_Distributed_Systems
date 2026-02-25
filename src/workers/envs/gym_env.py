"""Thin wrapper around standard Gymnasium envs for PPO validation.

Default: CartPole-v1.  Used to verify the PPO implementation works
on a known-good environment before debugging custom envs.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class GymEnvWrapper:
    """Adapter that gives any Gymnasium env a consistent interface."""

    def __init__(self, env_id: str = "CartPole-v1", seed: int | None = None):
        self.env = gym.make(env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._seed = seed

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        s = seed if seed is not None else self._seed
        obs, info = self.env.reset(seed=s)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info

    @property
    def spec(self) -> Any:
        return self.env.spec

    def close(self) -> None:
        self.env.close()
