"""Cost tracker: monitors compute units consumed by workers and training steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CostRecord:
    worker_id: str
    wall_time_s: float
    speed_factor: float
    num_transitions: int

    @property
    def compute_units(self) -> float:
        return self.wall_time_s * (1.0 / max(self.speed_factor, 0.01))


class CostTracker:
    """Tracks compute cost across workers and training steps."""

    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._step_times: list[float] = []
        self._peak_memory_bytes: int = 0

    def record_rollout(
        self,
        worker_id: str,
        wall_time_s: float,
        speed_factor: float,
        num_transitions: int,
    ) -> CostRecord:
        rec = CostRecord(
            worker_id=worker_id,
            wall_time_s=wall_time_s,
            speed_factor=speed_factor,
            num_transitions=num_transitions,
        )
        self._records.append(rec)
        return rec

    def record_step_time(self, duration_s: float) -> None:
        self._step_times.append(duration_s)

    def update_peak_memory(self) -> None:
        if torch.cuda.is_available():
            self._peak_memory_bytes = max(
                self._peak_memory_bytes, torch.cuda.max_memory_allocated()
            )

    @property
    def total_compute_units(self) -> float:
        return sum(r.compute_units for r in self._records)

    @property
    def total_trajectories(self) -> int:
        return len(self._records)

    @property
    def cost_per_trajectory(self) -> float:
        if not self._records:
            return 0.0
        return self.total_compute_units / len(self._records)

    @property
    def avg_step_time(self) -> float:
        if not self._step_times:
            return 0.0
        return sum(self._step_times) / len(self._step_times)

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_memory_bytes / (1024 * 1024)

    def summary(self) -> dict[str, Any]:
        return {
            "total_compute_units": self.total_compute_units,
            "total_trajectories": self.total_trajectories,
            "cost_per_trajectory": self.cost_per_trajectory,
            "avg_step_time_s": self.avg_step_time,
            "peak_memory_mb": self.peak_memory_mb,
        }
