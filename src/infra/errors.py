"""Typed exception hierarchy for the distributed RL platform.

Every failure mode gets a specific exception type so that callers can
handle each case explicitly and observability can label errors by kind.
"""

from __future__ import annotations


class DistRLError(Exception):
    """Base exception for all distributed RL platform errors."""


class WorkerFailureError(DistRLError):
    """A rollout worker actor died or raised an unrecoverable error."""

    def __init__(self, worker_id: str, reason: str = "unknown"):
        self.worker_id = worker_id
        self.reason = reason
        super().__init__(f"Worker {worker_id} failed: {reason}")


class RolloutTimeoutError(DistRLError):
    """A worker did not return a rollout within the configured deadline."""

    def __init__(self, worker_id: str, timeout_s: float):
        self.worker_id = worker_id
        self.timeout_s = timeout_s
        super().__init__(f"Worker {worker_id} timed out after {timeout_s:.1f}s")


class StaleRolloutError(DistRLError):
    """Trajectory was collected under an outdated policy version."""

    def __init__(self, expected_version: int, got_version: int):
        self.expected_version = expected_version
        self.got_version = got_version
        super().__init__(
            f"Stale rollout: expected policy_version>={expected_version}, got {got_version}"
        )


class CheckpointCorruptionError(DistRLError):
    """Checkpoint file is missing expected keys or fails integrity check."""

    def __init__(self, path: str, detail: str = ""):
        self.path = path
        self.detail = detail
        super().__init__(f"Corrupt checkpoint at {path}: {detail}")


class BufferOverflowError(DistRLError):
    """Batch collector is at capacity; backpressure should be engaged."""


class CoordinatorStateError(DistRLError):
    """Coordinator internal state is inconsistent."""
