"""Coordinator Ray actor: dispatches rollouts, monitors health, handles churn."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray

from src.coord.health import HealthMonitor
from src.coord.registry import WorkerRegistry

if TYPE_CHECKING:
    from src.infra.config import AppConfig

logger = logging.getLogger(__name__)


@ray.remote
class Coordinator:
    """Central coordinator that manages workers, dispatches rollouts, and tracks health."""

    def __init__(self, config: AppConfig, run_dir: str | None = None):
        self._config = config
        self._registry = WorkerRegistry()
        self._health = HealthMonitor(
            self._registry, timeout_s=config.coordinator.heartbeat_timeout_s
        )
        self._policy_version = 0
        self._current_weights: dict[str, Any] | None = None
        self._run_dir = Path(run_dir) if run_dir else None
        self._shutting_down = False

    def register_worker(
        self,
        worker_id: str,
        actor_handle: Any,
        speed_factor: float = 1.0,
        failure_rate: float = 0.0,
        max_batch: int = 512,
    ) -> dict[str, Any]:
        rec = self._registry.register(
            worker_id=worker_id,
            actor_handle=actor_handle,
            speed_factor=speed_factor,
            failure_rate=failure_rate,
            max_batch=max_batch,
        )
        logger.info("worker.registered", extra={"worker_id": worker_id})
        return {
            "worker_id": rec.worker_id,
            "policy_version": self._policy_version,
            "weights": self._current_weights,
        }

    def heartbeat(self, worker_id: str) -> bool:
        self._registry.heartbeat(worker_id)
        return True

    def report_rollout_complete(self, worker_id: str) -> None:
        self._registry.heartbeat(worker_id)
        self._registry.mark_idle(worker_id)

    def mark_worker_dead(self, worker_id: str) -> None:
        self._registry.mark_dead(worker_id)

    def update_policy(self, weights: dict[str, Any], version: int) -> None:
        self._current_weights = weights
        self._policy_version = version
        logger.info("policy.updated", extra={"version": version})

    def get_policy_version(self) -> int:
        return self._policy_version

    def get_policy_weights(self) -> dict[str, Any] | None:
        return self._current_weights

    def get_assignments(self, batch_fullness: float = 0.0) -> list[dict[str, Any]]:
        """Get rollout assignments for idle workers, respecting backpressure."""
        if self._shutting_down:
            return []

        if batch_fullness > 0.8:
            logger.debug("backpressure.engaged", extra={"fullness": batch_fullness})
            return []

        self._run_health_check()

        idle = self._registry.get_idle()
        assignments = []
        for rec in idle:
            self._registry.mark_busy(rec.worker_id)
            assignments.append(
                {
                    "worker_id": rec.worker_id,
                    "actor_handle": rec.actor_handle,
                    "rollout_steps": min(self._config.workers.rollout_steps, rec.max_batch),
                    "policy_version": self._policy_version,
                }
            )
        return assignments

    def get_status(self) -> dict[str, Any]:
        return {
            "policy_version": self._policy_version,
            "total_workers": self._registry.size,
            "healthy_workers": self._registry.healthy_count,
            "workers": self._registry.to_dict(),
        }

    def get_dead_workers(self) -> list[dict[str, Any]]:
        self._run_health_check()
        dead = self._registry.get_dead()
        return [{"worker_id": d.worker_id, "speed_factor": d.speed_factor} for d in dead]

    def remove_dead_workers(self) -> list[str]:
        """Remove dead workers from registry and return their IDs for replacement."""
        dead = self._registry.get_dead()
        removed = []
        for w in dead:
            self._registry.deregister(w.worker_id)
            removed.append(w.worker_id)
        return removed

    def shutdown(self) -> None:
        self._shutting_down = True
        logger.info("coordinator.shutdown")
        self._persist_state()

    def _run_health_check(self) -> None:
        newly_dead = self._health.check()
        if newly_dead:
            logger.warning(
                "workers.died",
                extra={"count": len(newly_dead), "worker_ids": newly_dead},
            )

    def _persist_state(self) -> None:
        if self._run_dir is None or not self._config.coordinator.persist_state:
            return
        state_file = self._run_dir / "coordinator_state.json"
        state = {
            "policy_version": self._policy_version,
            "workers": self._registry.to_dict(),
            "timestamp": time.time(),
        }
        try:
            tmp = state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(state_file)
        except Exception:
            logger.exception("coordinator.persist_failed")
