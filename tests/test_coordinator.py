"""Tests for coordinator, registry, and health monitor."""

from __future__ import annotations

import time

from src.coord.health import HealthMonitor
from src.coord.registry import WorkerRegistry, WorkerStatus


class TestWorkerRegistry:
    def test_register_and_lookup(self) -> None:
        reg = WorkerRegistry()
        rec = reg.register("w-0", actor_handle="fake_handle", speed_factor=0.5)
        assert rec.worker_id == "w-0"
        assert rec.speed_factor == 0.5
        assert reg.size == 1

    def test_deregister(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        reg.deregister("w-0")
        assert reg.size == 0

    def test_heartbeat(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        old_ts = reg.get_record("w-0").last_heartbeat  # type: ignore[union-attr]
        time.sleep(0.05)  # Windows timer resolution is ~15ms
        reg.heartbeat("w-0")
        new_ts = reg.get_record("w-0").last_heartbeat  # type: ignore[union-attr]
        assert new_ts >= old_ts

    def test_mark_dead(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        reg.mark_dead("w-0")
        assert reg.get_record("w-0").status == WorkerStatus.DEAD  # type: ignore[union-attr]
        assert reg.healthy_count == 0

    def test_get_idle(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        reg.register("w-1", actor_handle="h")
        reg.mark_busy("w-0")
        idle = reg.get_idle()
        assert len(idle) == 1
        assert idle[0].worker_id == "w-1"

    def test_check_timeouts(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        rec = reg.get_record("w-0")
        assert rec is not None
        rec.last_heartbeat = time.monotonic() - 100
        timed_out = reg.check_timeouts(timeout_s=10.0)
        assert "w-0" in timed_out

    def test_to_dict(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h", speed_factor=0.8)
        data = reg.to_dict()
        assert len(data) == 1
        assert data[0]["worker_id"] == "w-0"


class TestHealthMonitor:
    def test_detects_timeout(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        rec = reg.get_record("w-0")
        assert rec is not None
        rec.last_heartbeat = time.monotonic() - 100
        monitor = HealthMonitor(reg, timeout_s=10.0)
        dead = monitor.check()
        assert "w-0" in dead
        assert reg.get_record("w-0").status == WorkerStatus.DEAD  # type: ignore[union-attr]

    def test_healthy_worker_not_killed(self) -> None:
        reg = WorkerRegistry()
        reg.register("w-0", actor_handle="h")
        monitor = HealthMonitor(reg, timeout_s=100.0)
        dead = monitor.check()
        assert len(dead) == 0
        assert monitor.is_healthy
