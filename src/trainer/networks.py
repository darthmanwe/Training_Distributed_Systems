"""Policy and value networks for PPO.

Follows CleanRL conventions: orthogonal init, shared trunk option,
separate policy/value heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyValueNetwork(nn.Module):
    """MLP with separate policy and value heads for discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        critic_layers: list[nn.Module] = []
        actor_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            critic_layers.extend([_layer_init(nn.Linear(prev, h)), nn.Tanh()])
            actor_layers.extend([_layer_init(nn.Linear(prev, h)), nn.Tanh()])
            prev = h

        critic_layers.append(_layer_init(nn.Linear(prev, 1), std=1.0))
        actor_layers.append(_layer_init(nn.Linear(prev, act_dim), std=0.01))

        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return Categorical(logits=logits), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, entropy, value)."""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value
