"""Tests for the on-policy batch collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from src.infra.errors import StaleRolloutError
from src.trainer.batch_collector import RolloutBatchCollector
from src.workers.trajectory import Trajectory

if TYPE_CHECKING:
    from pathlib import Path


def _make_traj(length: int = 10, policy_version: int = 0, traj_id: str | None = None) -> Trajectory:
    t = Trajectory(
        observations=torch.randn(length, 4),
        actions=torch.randint(0, 2, (length,)),
        rewards=torch.randn(length),
        dones=torch.zeros(length),
        log_probs=torch.randn(length),
        values=torch.randn(length),
        policy_version=policy_version,
    )
    if traj_id:
        t.trajectory_id = traj_id
    return t


class TestRolloutBatchCollector:
    def test_add_and_ready(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=20, policy_version=0)
        bc.add(_make_traj(length=10))
        assert not bc.is_ready()
        bc.add(_make_traj(length=15))
        assert bc.is_ready()

    def test_consume_clears(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=10, policy_version=0)
        bc.add(_make_traj(length=15))
        batch = bc.consume()
        assert batch.size == 15
        assert bc.total_transitions == 0
        assert not bc.is_ready()

    def test_dedup(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=100, policy_version=0)
        t = _make_traj(length=5, traj_id="abc123")
        bc.add(t)
        bc.add(t)  # duplicate
        assert bc.total_transitions == 5  # not 10
        assert bc.dedup_hits == 1

    def test_stale_rejection(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=100, policy_version=5, staleness_window=1)
        with pytest.raises(StaleRolloutError):
            bc.add(_make_traj(length=5, policy_version=2))

    def test_fullness(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=100, policy_version=0)
        bc.add(_make_traj(length=50))
        assert abs(bc.fullness - 0.5) < 0.01

    def test_wal_and_recovery(self, tmp_output_dir: Path) -> None:
        wal_path = tmp_output_dir / "wal.jsonl"
        bc = RolloutBatchCollector(target_batch_size=100, policy_version=0, wal_path=wal_path)
        bc.add(_make_traj(length=5, traj_id="t1"))
        bc.add(_make_traj(length=5, traj_id="t2"))

        bc2 = RolloutBatchCollector(target_batch_size=100, policy_version=0, wal_path=wal_path)
        recovered = bc2.recover_from_wal()
        assert recovered == 2

    def test_cursor(self) -> None:
        bc = RolloutBatchCollector(target_batch_size=100, policy_version=3)
        bc.add(_make_traj(length=10, policy_version=3))
        cursor = bc.get_cursor()
        assert cursor["total_transitions"] == 10
        assert cursor["policy_version"] == 3
