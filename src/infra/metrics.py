"""OpenTelemetry metrics setup with Prometheus exporter."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_meter = None
_instruments: dict[str, Any] = {}


def setup_metrics(prometheus_port: int = 8000) -> dict[str, Any]:
    """Initialize OTel metrics with Prometheus exporter. Returns instrument dict."""
    global _meter, _instruments

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import start_http_server

        start_http_server(prometheus_port)

        resource = Resource.create({"service.name": "dist-rl-platform"})
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _meter = provider.get_meter("dist_rl")

        _instruments = {
            "rollout_throughput": _meter.create_counter(
                "dist_rl.rollout.throughput",
                description="Total trajectories received",
                unit="trajectories",
            ),
            "rollout_latency": _meter.create_histogram(
                "dist_rl.rollout.latency",
                description="Time from dispatch to receipt",
                unit="seconds",
            ),
            "trainer_step_duration": _meter.create_histogram(
                "dist_rl.trainer.step_duration",
                description="Wall time per training step",
                unit="seconds",
            ),
            "trainer_reward": _meter.create_histogram(
                "dist_rl.trainer.reward",
                description="Reward per evaluation",
            ),
            "trainer_kl": _meter.create_histogram(
                "dist_rl.trainer.kl_divergence",
                description="Approximate KL divergence",
            ),
            "buffer_depth": _meter.create_up_down_counter(
                "dist_rl.buffer.depth",
                description="Current batch fill level",
            ),
            "buffer_dedup_hits": _meter.create_counter(
                "dist_rl.buffer.dedup_hits",
                description="Deduplicated trajectory count",
            ),
            "worker_heartbeat_latency": _meter.create_histogram(
                "dist_rl.worker.heartbeat_latency",
                description="Heartbeat round-trip time",
                unit="seconds",
            ),
            "worker_failure_count": _meter.create_counter(
                "dist_rl.worker.failure_count",
                description="Worker failures",
            ),
            "worker_active_count": _meter.create_up_down_counter(
                "dist_rl.worker.active_count",
                description="Currently active workers",
            ),
            "cost_total": _meter.create_counter(
                "dist_rl.cost.total_compute_units",
                description="Total compute units consumed",
            ),
            "checkpoint_duration": _meter.create_histogram(
                "dist_rl.checkpoint.duration",
                description="Checkpoint save duration",
                unit="seconds",
            ),
        }

        logger.info(f"Metrics server started on :{prometheus_port}/metrics")

    except Exception as e:
        logger.warning(f"Failed to initialize metrics: {e}. Continuing without metrics.")
        _instruments = {}

    return _instruments


def record(name: str, value: float, attributes: dict[str, str] | None = None) -> None:
    """Record a metric value. No-op if metrics not initialized."""
    inst = _instruments.get(name)
    if inst is None:
        return
    attrs = attributes or {}
    if hasattr(inst, "add"):
        inst.add(value, attrs)
    elif hasattr(inst, "record"):
        inst.record(value, attrs)
