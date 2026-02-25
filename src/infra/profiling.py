"""Performance profiling utilities wrapping PyTorch profiler."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def profiler_context(
    run_dir: Path,
    step: int,
    enabled: bool = False,
) -> Generator[None, None, None]:
    """Context manager that optionally profiles a training step."""
    if not enabled:
        yield
        return

    prof_dir = run_dir / "profiling"
    prof_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        yield

    trace_file = prof_dir / f"trace_step_{step}.json"
    prof.export_chrome_trace(str(trace_file))

    key_avgs = prof.key_averages()
    summary_file = prof_dir / f"summary_step_{step}.txt"
    with open(summary_file, "w") as f:
        f.write(key_avgs.table(sort_by="cpu_time_total", row_limit=30))

    logger.info(
        "profiler.saved",
        extra={"step": step, "trace": str(trace_file), "summary": str(summary_file)},
    )
