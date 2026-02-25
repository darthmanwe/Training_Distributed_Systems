"""Trajectory data structures for rollout collection and training."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import torch


@dataclass
class Trajectory:
    """A single rollout segment from one worker."""

    observations: torch.Tensor  # (T, obs_dim)
    actions: torch.Tensor  # (T,)
    rewards: torch.Tensor  # (T,)
    dones: torch.Tensor  # (T,) — True at terminal states
    log_probs: torch.Tensor  # (T,)
    values: torch.Tensor  # (T,)
    truncated: bool = False  # True if rollout ended by step limit, not episode end
    last_value: float = 0.0  # bootstrap value for truncation (V(s_{T+1}))
    policy_version: int = 0
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    worker_id: str = ""

    @property
    def length(self) -> int:
        return self.observations.shape[0]


@dataclass
class TrajectoryBatch:
    """Flattened batch of trajectories ready for PPO training."""

    observations: torch.Tensor  # (B, obs_dim)
    actions: torch.Tensor  # (B,)
    rewards: torch.Tensor  # (B,)
    dones: torch.Tensor  # (B,)
    log_probs: torch.Tensor  # (B,)
    values: torch.Tensor  # (B,)
    advantages: torch.Tensor  # (B,) — filled during GAE computation
    returns: torch.Tensor  # (B,) — filled during GAE computation

    @property
    def size(self) -> int:
        return self.observations.shape[0]

    def to_device(self, device: torch.device) -> TrajectoryBatch:
        return TrajectoryBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )

    def minibatch_indices(self, num_minibatches: int) -> list[torch.Tensor]:
        """Return shuffled index splits for minibatch iteration."""
        indices = torch.randperm(self.size)
        return list(torch.chunk(indices, num_minibatches))
