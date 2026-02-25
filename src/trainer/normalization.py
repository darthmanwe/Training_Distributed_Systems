"""Running mean/std normalization for observations and rewards.

Uses Welford's online algorithm. State is checkpointable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self._epsilon = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1 and self.mean.ndim > 0:
            batch = batch.reshape(1, -1)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0] if batch.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]


class ObservationNormalizer:
    """Normalizes observations using running statistics, clips to [-clip, clip]."""

    def __init__(self, obs_dim: int, clip: float = 10.0):
        self._rms = RunningMeanStd(shape=(obs_dim,))
        self._clip = clip

    def normalize(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        obs_np = obs.cpu().numpy()
        if update:
            self._rms.update(obs_np)
        mean = torch.tensor(self._rms.mean, dtype=torch.float32, device=obs.device)
        std = torch.tensor(np.sqrt(self._rms.var + 1e-8), dtype=torch.float32, device=obs.device)
        normalized = (obs - mean) / std
        return torch.clamp(normalized, -self._clip, self._clip)

    def state_dict(self) -> dict[str, Any]:
        return self._rms.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._rms.load_state_dict(state)


class RewardNormalizer:
    """Normalizes rewards by running std of discounted returns."""

    def __init__(self, gamma: float = 0.99):
        self._rms = RunningMeanStd(shape=())
        self._gamma = gamma
        self._ret = 0.0

    def normalize(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        rew_np = rewards.cpu().numpy()
        done_np = dones.cpu().numpy()
        normalized = np.zeros_like(rew_np)
        for i in range(len(rew_np)):
            self._ret = self._ret * self._gamma * (1 - done_np[i]) + rew_np[i]
            self._rms.update(np.array([self._ret]))
            normalized[i] = rew_np[i] / (np.sqrt(self._rms.var) + 1e-8)
        return torch.tensor(normalized, dtype=torch.float32, device=rewards.device)

    def state_dict(self) -> dict[str, Any]:
        return {"rms": self._rms.state_dict(), "ret": self._ret}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._rms.load_state_dict(state["rms"])
        self._ret = state["ret"]
