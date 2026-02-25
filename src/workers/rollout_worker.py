"""Rollout worker Ray actor: collects trajectories from an environment."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import ray
import torch
import torch.nn as nn

from src.infra.errors import WorkerFailureError
from src.infra.seeding import seed_everything, worker_seed
from src.workers.envs.gym_env import GymEnvWrapper
from src.workers.envs.scheduling_env import JobSchedulingEnv
from src.workers.trajectory import Trajectory

if TYPE_CHECKING:
    from src.infra.config import AppConfig

logger = logging.getLogger(__name__)


def _make_env(config: AppConfig, seed: int) -> JobSchedulingEnv | GymEnvWrapper:
    if config.env.name == "cartpole":
        return GymEnvWrapper(env_id="CartPole-v1", seed=seed)
    return JobSchedulingEnv(
        num_env_workers=config.env.num_env_workers,
        num_jobs=config.env.num_jobs,
        max_queue_len=config.env.max_queue_len,
        max_steps_per_episode=config.env.max_steps_per_episode,
        seed=seed,
    )


@ray.remote
class RolloutWorker:
    """Collects rollout trajectories with configurable heterogeneity simulation."""

    def __init__(
        self,
        worker_id: str,
        config: AppConfig,
        speed_factor: float = 1.0,
        failure_rate: float = 0.0,
        max_batch: int = 512,
    ):
        self._worker_id = worker_id
        self._config = config
        self._speed_factor = speed_factor
        self._failure_rate = failure_rate
        self._max_batch = max_batch

        wseed = worker_seed(config.seed, int(worker_id.split("-")[-1]))
        seed_everything(wseed)
        self._rng = np.random.default_rng(wseed)

        self._env = _make_env(config, wseed)
        self._policy: nn.Module | None = None
        self._policy_version = 0
        self._obs: np.ndarray | None = None
        self._reset_env()

    def _reset_env(self) -> None:
        self._obs, _ = self._env.reset()

    def set_policy(self, state_dict: dict[str, Any], version: int) -> None:
        if self._policy is not None:
            self._policy.load_state_dict(state_dict)
            self._policy.eval()
        self._policy_version = version

    def set_policy_module(self, policy: nn.Module, version: int) -> None:
        """Set the full policy module (used on first init)."""
        self._policy = policy
        self._policy.eval()
        self._policy_version = version

    def get_worker_id(self) -> str:
        return self._worker_id

    @torch.no_grad()
    def collect_rollout(self, num_steps: int) -> Trajectory:
        """Run the environment for num_steps and return a trajectory."""
        if self._rng.random() < self._failure_rate:
            raise WorkerFailureError(self._worker_id, "simulated_failure")

        actual_steps = min(num_steps, self._max_batch)
        if self._policy is None:
            raise WorkerFailureError(self._worker_id, "policy_not_set")

        obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []

        obs = self._obs
        truncated_at_end = False
        last_value = 0.0

        for _ in range(actual_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, value = self._policy(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs_list.append(obs)
            act_list.append(action.item())
            logp_list.append(log_prob.item())
            val_list.append(value.item())

            next_obs, reward, terminated, truncated, _info = self._env.step(action.item())

            rew_list.append(reward)
            done_list.append(float(terminated))

            if terminated or truncated:
                if truncated and not terminated:
                    truncated_at_end = True
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                    _, lv = self._policy(next_obs_t)
                    last_value = lv.item()
                next_obs, _ = self._env.reset()

            obs = next_obs

            if self._speed_factor < 1.0:
                delay = (1.0 - self._speed_factor) * 0.01
                time.sleep(delay)

        self._obs = obs

        return Trajectory(
            observations=torch.tensor(np.array(obs_list), dtype=torch.float32),
            actions=torch.tensor(act_list, dtype=torch.long),
            rewards=torch.tensor(rew_list, dtype=torch.float32),
            dones=torch.tensor(done_list, dtype=torch.float32),
            log_probs=torch.tensor(logp_list, dtype=torch.float32),
            values=torch.tensor(val_list, dtype=torch.float32),
            truncated=truncated_at_end,
            last_value=last_value,
            policy_version=self._policy_version,
            worker_id=self._worker_id,
        )

    def ping(self) -> str:
        return self._worker_id

    def shutdown(self) -> None:
        logger.info("worker.shutdown", extra={"worker_id": self._worker_id})
        ray.actor.exit_actor()
