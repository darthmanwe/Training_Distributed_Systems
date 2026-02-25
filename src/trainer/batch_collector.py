"""On-policy rollout batch collector with dedup and WAL.

NOT a replay buffer. Accumulates trajectories under the current policy,
then flushes entirely after the trainer consumes the batch.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import torch

from src.infra.errors import BufferOverflowError, StaleRolloutError
from src.workers.trajectory import Trajectory, TrajectoryBatch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class RolloutBatchCollector:
    """Accumulates on-policy trajectories until a batch is ready for training."""

    def __init__(
        self,
        target_batch_size: int,
        policy_version: int = 0,
        staleness_window: int = 1,
        wal_path: Path | None = None,
    ):
        self._target = target_batch_size
        self._policy_version = policy_version
        self._staleness_window = staleness_window
        self._trajectories: list[Trajectory] = []
        self._seen_ids: set[str] = set()
        self._total_transitions = 0
        self._dedup_hits = 0

        self._wal_path = wal_path
        if wal_path:
            wal_path.parent.mkdir(parents=True, exist_ok=True)

    def set_policy_version(self, version: int) -> None:
        self._policy_version = version

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory, checking version and dedup."""
        if trajectory.trajectory_id in self._seen_ids:
            self._dedup_hits += 1
            logger.debug("dedup.hit", extra={"trajectory_id": trajectory.trajectory_id})
            return

        min_version = self._policy_version - self._staleness_window
        if trajectory.policy_version < min_version:
            raise StaleRolloutError(min_version, trajectory.policy_version)

        if self._total_transitions >= self._target * 1.5:
            raise BufferOverflowError(
                f"Collector at {self._total_transitions}/{self._target} transitions"
            )

        self._trajectories.append(trajectory)
        self._seen_ids.add(trajectory.trajectory_id)
        self._total_transitions += trajectory.length

        if self._wal_path:
            self._wal_append(trajectory)

    def is_ready(self) -> bool:
        return self._total_transitions >= self._target

    @property
    def fullness(self) -> float:
        if self._target <= 0:
            return 0.0
        return min(1.0, self._total_transitions / self._target)

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def dedup_hits(self) -> int:
        return self._dedup_hits

    def consume(self) -> TrajectoryBatch:
        """Build a TrajectoryBatch and clear the collector (on-policy: data is discarded)."""
        if not self._trajectories:
            raise ValueError("No trajectories to consume")

        all_obs = torch.cat([t.observations for t in self._trajectories], dim=0)
        all_act = torch.cat([t.actions for t in self._trajectories], dim=0)
        all_rew = torch.cat([t.rewards for t in self._trajectories], dim=0)
        all_done = torch.cat([t.dones for t in self._trajectories], dim=0)
        all_logp = torch.cat([t.log_probs for t in self._trajectories], dim=0)
        all_val = torch.cat([t.values for t in self._trajectories], dim=0)

        batch = TrajectoryBatch(
            observations=all_obs,
            actions=all_act,
            rewards=all_rew,
            dones=all_done,
            log_probs=all_logp,
            values=all_val,
            advantages=torch.zeros_like(all_rew),
            returns=torch.zeros_like(all_rew),
        )

        self._clear()

        if self._wal_path and self._wal_path.exists():
            self._wal_path.unlink()

        return batch

    @property
    def trajectories(self) -> list[Trajectory]:
        """Access raw trajectories for GAE computation before consume()."""
        return self._trajectories

    def _clear(self) -> None:
        self._trajectories.clear()
        self._seen_ids.clear()
        self._total_transitions = 0

    def _wal_append(self, traj: Trajectory) -> None:
        if self._wal_path is None:
            return
        entry = {
            "trajectory_id": traj.trajectory_id,
            "policy_version": traj.policy_version,
            "worker_id": traj.worker_id,
            "num_transitions": traj.length,
        }
        try:
            with open(self._wal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.exception("wal.append_failed")

    def recover_from_wal(self) -> int:
        """Replay WAL to rebuild seen_ids set. Returns number of entries recovered."""
        if self._wal_path is None or not self._wal_path.exists():
            return 0
        count = 0
        with open(self._wal_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self._seen_ids.add(entry["trajectory_id"])
                count += 1
        return count

    def get_cursor(self) -> dict:
        """Snapshot for checkpointing."""
        return {
            "total_transitions": self._total_transitions,
            "seen_ids": list(self._seen_ids),
            "policy_version": self._policy_version,
        }
