"""Custom Gymnasium environment: job scheduling across heterogeneous workers.

The agent assigns incoming jobs to workers to minimise weighted latency + cost.
Observation encodes the job queue state and per-worker load.  Action selects
which worker receives the next job (or "wait").
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JobSchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_env_workers: int = 5,
        num_jobs: int = 20,
        max_queue_len: int = 10,
        max_steps_per_episode: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.num_env_workers = num_env_workers
        self.num_jobs = num_jobs
        self.max_queue_len = max_queue_len
        self.max_steps = max_steps_per_episode

        self.job_features = 3  # size, priority, age
        self.worker_features = 3  # load, speed, queue_depth
        obs_dim = max_queue_len * self.job_features + num_env_workers * self.worker_features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_env_workers + 1)  # +1 for "wait"

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._total_latency = 0.0
        self._total_cost = 0.0
        self._jobs_completed = 0

        self._job_queue: list[dict[str, float]] = []
        self._worker_speeds: np.ndarray = np.array([])
        self._worker_loads: np.ndarray = np.array([])
        self._worker_queues: np.ndarray = np.array([])
        self._jobs_remaining = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._worker_speeds = self._rng.uniform(0.3, 1.0, size=self.num_env_workers).astype(
            np.float32
        )
        self._worker_loads = np.zeros(self.num_env_workers, dtype=np.float32)
        self._worker_queues = np.zeros(self.num_env_workers, dtype=np.float32)

        self._job_queue = []
        for _ in range(min(self.max_queue_len, self.num_jobs)):
            self._job_queue.append(self._make_job())

        self._jobs_remaining = self.num_jobs - len(self._job_queue)
        self._step_count = 0
        self._total_latency = 0.0
        self._total_cost = 0.0
        self._jobs_completed = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        reward = 0.0

        if action < self.num_env_workers and len(self._job_queue) > 0:
            job = self._job_queue.pop(0)
            worker_idx = action

            latency = job["size"] / (self._worker_speeds[worker_idx] + 1e-6)
            cost = latency * (1.0 + self._worker_loads[worker_idx])

            self._worker_loads[worker_idx] = min(
                1.0, self._worker_loads[worker_idx] + job["size"] * 0.3
            )
            self._worker_queues[worker_idx] = min(
                1.0, self._worker_queues[worker_idx] + 0.1
            )

            self._total_latency += latency
            self._total_cost += cost
            self._jobs_completed += 1

            reward = -0.5 * latency - 0.3 * cost + 0.2 * job["priority"]

            if self._jobs_remaining > 0:
                self._job_queue.append(self._make_job())
                self._jobs_remaining -= 1

        self._worker_loads *= 0.95
        self._worker_queues *= 0.95

        terminated = len(self._job_queue) == 0 and self._jobs_remaining == 0
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _make_job(self) -> dict[str, float]:
        return {
            "size": float(self._rng.uniform(0.1, 1.0)),
            "priority": float(self._rng.uniform(0.0, 1.0)),
            "age": 0.0,
        }

    def _get_obs(self) -> np.ndarray:
        job_obs = np.zeros(self.max_queue_len * self.job_features, dtype=np.float32)
        for i, job in enumerate(self._job_queue[: self.max_queue_len]):
            base = i * self.job_features
            job_obs[base] = job["size"]
            job_obs[base + 1] = job["priority"]
            job_obs[base + 2] = min(1.0, job["age"] / 50.0)

        worker_obs = np.zeros(self.num_env_workers * self.worker_features, dtype=np.float32)
        for i in range(self.num_env_workers):
            base = i * self.worker_features
            worker_obs[base] = self._worker_loads[i]
            worker_obs[base + 1] = self._worker_speeds[i]
            worker_obs[base + 2] = self._worker_queues[i]

        return np.concatenate([job_obs, worker_obs])

    def _get_info(self) -> dict[str, Any]:
        return {
            "jobs_completed": self._jobs_completed,
            "jobs_remaining": self._jobs_remaining + len(self._job_queue),
            "total_latency": self._total_latency,
            "total_cost": self._total_cost,
            "step": self._step_count,
        }
