"""Health monitor: checks heartbeat staleness and marks dead workers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.coord.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Checks worker heartbeats and marks stale workers as dead."""

    def __init__(self, registry: WorkerRegistry, timeout_s: float = 15.0):
        self._registry = registry
        self._timeout_s = timeout_s

    def check(self) -> list[str]:
        """Run one health check cycle. Returns list of newly-dead worker IDs."""
        timed_out = self._registry.check_timeouts(self._timeout_s)
        for wid in timed_out:
            self._registry.mark_dead(wid)
            logger.warning("worker.timeout", extra={"worker_id": wid})
        return timed_out

    @property
    def is_healthy(self) -> bool:
        return self._registry.healthy_count > 0
