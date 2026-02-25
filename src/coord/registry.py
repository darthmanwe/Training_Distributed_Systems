"""Worker registry: tracks all known rollout workers, their capabilities, and status."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkerStatus(StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    DEAD = "dead"
    DRAINING = "draining"


@dataclass
class WorkerRecord:
    worker_id: str
    actor_handle: Any
    speed_factor: float = 1.0
    failure_rate: float = 0.0
    max_batch: int = 512
    status: WorkerStatus = WorkerStatus.IDLE
    last_heartbeat: float = field(default_factory=time.monotonic)
    assigned_policy_version: int = 0
    pending_rollout_ids: list[str] = field(default_factory=list)


class WorkerRegistry:
    """In-memory registry of rollout workers with capacity-aware selection."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}

    def register(
        self,
        worker_id: str,
        actor_handle: Any,
        speed_factor: float = 1.0,
        failure_rate: float = 0.0,
        max_batch: int = 512,
    ) -> WorkerRecord:
        rec = WorkerRecord(
            worker_id=worker_id,
            actor_handle=actor_handle,
            speed_factor=speed_factor,
            failure_rate=failure_rate,
            max_batch=max_batch,
        )
        self._workers[worker_id] = rec
        return rec

    def deregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def heartbeat(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].last_heartbeat = time.monotonic()

    def mark_dead(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].status = WorkerStatus.DEAD

    def mark_busy(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].status = WorkerStatus.BUSY

    def mark_idle(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].status = WorkerStatus.IDLE

    def get_healthy(self) -> list[WorkerRecord]:
        return [
            w for w in self._workers.values() if w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY)
        ]

    def get_idle(self) -> list[WorkerRecord]:
        return [w for w in self._workers.values() if w.status == WorkerStatus.IDLE]

    def get_dead(self) -> list[WorkerRecord]:
        return [w for w in self._workers.values() if w.status == WorkerStatus.DEAD]

    def get_record(self, worker_id: str) -> WorkerRecord | None:
        return self._workers.get(worker_id)

    def get_all(self) -> list[WorkerRecord]:
        return list(self._workers.values())

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def healthy_count(self) -> int:
        return len(self.get_healthy())

    def check_timeouts(self, timeout_s: float) -> list[str]:
        """Return worker_ids that have exceeded the heartbeat timeout."""
        now = time.monotonic()
        timed_out = []
        for w in self._workers.values():
            if w.status != WorkerStatus.DEAD and (now - w.last_heartbeat) > timeout_s:
                timed_out.append(w.worker_id)
        return timed_out

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "worker_id": w.worker_id,
                "speed_factor": w.speed_factor,
                "failure_rate": w.failure_rate,
                "max_batch": w.max_batch,
                "status": w.status.value,
                "last_heartbeat": w.last_heartbeat,
            }
            for w in self._workers.values()
        ]
