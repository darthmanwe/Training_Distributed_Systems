"""Evaluation harness: runs the policy greedily and reports metrics."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from src.workers.envs.gym_env import GymEnvWrapper
from src.workers.envs.scheduling_env import JobSchedulingEnv

if TYPE_CHECKING:
    from pathlib import Path

    from src.infra.config import AppConfig

logger = logging.getLogger(__name__)


def make_eval_env(config: AppConfig, seed: int) -> JobSchedulingEnv | GymEnvWrapper:
    if config.env.name == "cartpole":
        return GymEnvWrapper(env_id="CartPole-v1", seed=seed)
    return JobSchedulingEnv(
        num_env_workers=config.env.num_env_workers,
        num_jobs=config.env.num_jobs,
        max_queue_len=config.env.max_queue_len,
        max_steps_per_episode=config.env.max_steps_per_episode,
        seed=seed,
    )


class Evaluator:
    """Runs periodic greedy evaluation and logs results."""

    def __init__(self, config: AppConfig, run_dir: Path | None = None):
        self._config = config
        self._run_dir = run_dir
        self._eval_seed = config.seed + 10000

    @torch.no_grad()
    def evaluate(
        self, policy: nn.Module, step: int, num_episodes: int | None = None
    ) -> dict[str, Any]:
        """Run policy greedily for num_episodes. Returns aggregated metrics."""
        n_eps = num_episodes or self._config.trainer.eval_episodes
        env = make_eval_env(self._config, self._eval_seed)
        policy.eval()

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for ep in range(n_eps):
            obs, _ = env.reset(seed=self._eval_seed + ep)
            total_reward = 0.0
            length = 0
            done = False

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist, _ = policy(obs_t)
                action = dist.probs.argmax(dim=-1).item()  # greedy
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                length += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        result = {
            "step": step,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "num_episodes": n_eps,
        }

        if self._run_dir:
            eval_dir = self._run_dir / "eval"
            eval_dir.mkdir(exist_ok=True)
            with open(eval_dir / f"step_{step}.json", "w") as f:
                json.dump(result, f, indent=2)

        logger.info(
            "eval.complete",
            extra={
                "step": step,
                "mean_reward": result["mean_reward"],
                "mean_length": result["mean_length"],
            },
        )

        return result
