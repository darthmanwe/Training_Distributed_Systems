"""PPO trainer following CleanRL's '37 implementation details'.

Supports: GAE with truncation bootstrap, clipped surrogate, clipped value loss,
entropy bonus, advantage normalization, gradient clipping, linear LR decay,
observation normalization, reward normalization, early stopping on KL.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from src.infra.seeding import capture_rng_state, restore_rng_state
from src.trainer.networks import PolicyValueNetwork
from src.trainer.normalization import ObservationNormalizer, RewardNormalizer
from src.workers.trajectory import Trajectory, TrajectoryBatch

if TYPE_CHECKING:
    from src.infra.config import TrainerConfig

logger = logging.getLogger(__name__)


def compute_gae(
    trajectories: list[Trajectory],
    gamma: float,
    gae_lambda: float,
) -> TrajectoryBatch:
    """Compute GAE advantages and returns for a list of trajectories.

    Handles truncation bootstrap: when a trajectory is truncated (not terminated),
    the last_value is used as the bootstrap target instead of 0.
    """
    all_obs, all_act, all_rew, all_done, all_logp, all_val = [], [], [], [], [], []
    all_adv, all_ret = [], []

    for traj in trajectories:
        traj_len = traj.length
        advantages = torch.zeros(traj_len, dtype=torch.float32)
        last_gae = 0.0

        # Bootstrap value: use last_value if truncated, else 0
        next_value = traj.last_value if traj.truncated else 0.0

        for t in reversed(range(traj_len)):
            if t == traj_len - 1:
                next_nonterminal = 1.0 - traj.dones[t].item()
                if traj.truncated:
                    next_nonterminal = 1.0
                next_val = next_value
            else:
                next_nonterminal = 1.0 - traj.dones[t].item()
                next_val = traj.values[t + 1].item()

            delta = traj.rewards[t].item() + gamma * next_val * next_nonterminal - traj.values[t].item()
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + traj.values

        all_obs.append(traj.observations)
        all_act.append(traj.actions)
        all_rew.append(traj.rewards)
        all_done.append(traj.dones)
        all_logp.append(traj.log_probs)
        all_val.append(traj.values)
        all_adv.append(advantages)
        all_ret.append(returns)

    return TrajectoryBatch(
        observations=torch.cat(all_obs),
        actions=torch.cat(all_act),
        rewards=torch.cat(all_rew),
        dones=torch.cat(all_done),
        log_probs=torch.cat(all_logp),
        values=torch.cat(all_val),
        advantages=torch.cat(all_adv),
        returns=torch.cat(all_ret),
    )


class PPOTrainer:
    """PPO with clipped objective, value clipping, entropy bonus, and LR decay."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TrainerConfig,
        device: torch.device | None = None,
    ):
        self._config = config
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_step = 0

        self._network = PolicyValueNetwork(obs_dim, act_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=config.lr, eps=1e-5)

        total_updates = config.total_timesteps // config.batch_size
        self._scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(total_updates, 1),
        )

        self._obs_normalizer: ObservationNormalizer | None = None
        if config.normalize_obs:
            self._obs_normalizer = ObservationNormalizer(obs_dim)

        self._reward_normalizer: RewardNormalizer | None = None
        if config.normalize_reward:
            self._reward_normalizer = RewardNormalizer(gamma=config.gamma)

    @property
    def policy_module(self) -> nn.Module:
        return self._network

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def device(self) -> torch.device:
        return self._device

    def get_policy_state_dict(self) -> dict[str, Any]:
        return {k: v.cpu() for k, v in self._network.state_dict().items()}

    def load_policy_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._network.load_state_dict(state_dict)

    def get_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "model": self._network.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "global_step": self._global_step,
            "rng": capture_rng_state(),
        }
        if self._obs_normalizer:
            state["obs_normalizer"] = self._obs_normalizer.state_dict()
        if self._reward_normalizer:
            state["reward_normalizer"] = self._reward_normalizer.state_dict()
        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._network.load_state_dict(state["model"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        self._global_step = state["global_step"]
        restore_rng_state(state["rng"])
        if self._obs_normalizer and "obs_normalizer" in state:
            self._obs_normalizer.load_state_dict(state["obs_normalizer"])
        if self._reward_normalizer and "reward_normalizer" in state:
            self._reward_normalizer.load_state_dict(state["reward_normalizer"])

    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        """Run PPO update on a batch. Returns metrics dictionary."""
        self._global_step += 1
        cfg = self._config

        batch = batch.to_device(self._device)

        # Normalize observations
        if self._obs_normalizer:
            batch = TrajectoryBatch(
                observations=self._obs_normalizer.normalize(batch.observations),
                actions=batch.actions,
                rewards=batch.rewards,
                dones=batch.dones,
                log_probs=batch.log_probs,
                values=batch.values,
                advantages=batch.advantages,
                returns=batch.returns,
            )

        clip_fracs = []
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _epoch in range(cfg.num_epochs):
            mb_indices = batch.minibatch_indices(cfg.num_minibatches)
            for indices in mb_indices:
                mb_obs = batch.observations[indices]
                mb_act = batch.actions[indices]
                mb_logp = batch.log_probs[indices]
                mb_adv = batch.advantages[indices]
                mb_ret = batch.returns[indices]
                mb_val = batch.values[indices]

                # Normalize advantages per minibatch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_logp, entropy, new_val = self._network.get_action_and_value(
                    mb_obs, mb_act
                )

                log_ratio = new_logp - mb_logp
                ratio = torch.exp(log_ratio)

                # Approximate KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                    clip_fracs.append(clip_frac)

                # Clipped surrogate objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss
                v_loss_unclipped = (new_val - mb_ret) ** 2
                v_clipped = mb_val + torch.clamp(
                    new_val - mb_val, -cfg.clip_eps, cfg.clip_eps
                )
                v_loss_clipped = (v_clipped - mb_ret) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._network.parameters(), cfg.max_grad_norm)
                self._optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl
                n_updates += 1

            # Early stopping on KL
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                logger.info(
                    "ppo.early_stop",
                    extra={"epoch": _epoch, "approx_kl": approx_kl, "target": cfg.target_kl},
                )
                break

        self._scheduler.step()

        # Explained variance
        with torch.no_grad():
            y_pred = batch.values.cpu().numpy()
            y_true = batch.returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-8 else 0.0

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": float(explained_var),
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            "global_step": self._global_step,
        }
