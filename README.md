# Distributed RL Platform

A fault-tolerant distributed ML system that trains a policy using reinforcement learning while handling **heterogeneous workers**, **worker churn**, **checkpoint recovery**, and **reproducible experiments**. Built to demonstrate "research-to-production" discipline: orchestration, fault tolerance, observability, and performance tuning.

## Why this exists

Most ML demos ignore the hard parts: flaky workers, partial failures, throughput imbalance, and reproducibility. This project focuses on the systems layer around ML/RL training:
- Distributed rollout collection across heterogeneous workers
- Centralized training loop (PPO) with on-policy data management
- Durable buffering + dedup to survive retries
- Checkpointing + recovery that actually works
- Observability (throughput, latency, worker health, RL diagnostics)
- Repeatable experiments with config snapshots and deterministic seeding

## Architecture

```
           +-------------------+
           |    Coordinator    |
           | registry + health |
           +---------+---------+
                     |
          assignments|heartbeats
                     |
     +---------------+----------------+
     |               |                |
+----v-----+   +-----v------+   +----v-----+
| Worker 0 |   |  Worker 1  |   | Worker N |
| spd=1.0  |   |  spd=0.5   |   | spd=0.3  |
| fail=0%  |   |  fail=10%  |   | fail=20% |
+----+-----+   +-----+------+   +----+-----+
     | trajectories   |               |
     +--------+-------+-------+-------+
              |               |
       +------v------+       |
       |    Batch     |<------+
       |  Collector   |  (on-policy, dedup, WAL)
       +------+------+
              |
       +------v------+
       |   PPO       |
       |  Trainer    |
       +------+------+
              |
       +------v------+
       | Checkpoints |
       +-------------+
```

### Key data flow

1. Coordinator broadcasts policy weights to all workers
2. Workers collect rollouts asynchronously (different speeds, failure rates)
3. Batch collector accumulates on-policy data with dedup and version checking
4. Trainer runs PPO update when batch is full, then discards data (on-policy)
5. Repeat with updated policy

## Quickstart

### Install

```bash
# Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -e ".[dev]"
```

### Run locally

```bash
# CartPole validation (fast, verifies PPO works)
python scripts/run_local.py --config configs/cartpole.yaml

# Full scheduling environment
python scripts/run_local.py --config configs/base.yaml

# With worker churn (kills workers randomly)
python scripts/run_local.py --config configs/churn.yaml

# Resume from checkpoint
python scripts/run_local.py --config configs/base.yaml --resume outputs/<run_id>/checkpoints/latest.pt
```

### Run with observability stack

```bash
# Start Prometheus + Grafana
docker compose -f docker/docker-compose.yaml up -d

# Run training (metrics exported on :8000)
python scripts/run_local.py --config configs/base.yaml

# Dashboard at http://localhost:3000 (admin/admin)
```

## Config-driven experiments

Runs are fully config-driven. Each run writes:
- `outputs/<run_id>/config_resolved.yaml` -- full config snapshot
- `outputs/<run_id>/run_metadata.json` -- git SHA, versions, platform
- `outputs/<run_id>/metrics.jsonl` -- training and eval metrics
- `outputs/<run_id>/checkpoints/` -- model + optimizer + RNG state
- `outputs/<run_id>/eval/` -- per-step evaluation results
- `outputs/<run_id>/logs/` -- structured JSON logs

Override any config value from CLI:

```bash
python scripts/run_local.py --config configs/base.yaml trainer.lr=1e-3 seed=123 workers.num_workers=8
```

## Fault tolerance model

- Workers can fail mid-rollout (simulated via `failure_rate` config)
- Coordinator detects dead workers via heartbeat timeout
- Dead workers are automatically replaced with fresh actors
- Trainer deduplicates trajectories by `trajectory_id`
- Checkpoints include: model + optimizer + scheduler + RNG state + normalizer stats
- Runs can resume exactly from checkpoint

## Metrics and observability

| Metric | Type | Description |
|--------|------|-------------|
| `dist_rl.rollout.throughput` | Counter | Trajectories received |
| `dist_rl.trainer.step_duration` | Histogram | Training step wall time |
| `dist_rl.trainer.reward` | Histogram | Evaluation reward |
| `dist_rl.worker.failure_count` | Counter | Worker failures by ID |
| `dist_rl.buffer.depth` | Gauge | Current batch fill level |
| `dist_rl.worker.active_count` | Gauge | Active worker count |

## Tech stack

| Component | Technology | Why |
|-----------|------------|-----|
| Training | PyTorch | Industry standard, compile/AMP support |
| Orchestration | Ray | Actor model, built-in fault detection, dashboard |
| RL Algorithm | PPO (CleanRL-style) | Well-understood, stable, 37 implementation details |
| Environment | Gymnasium | Standard interface, custom + CartPole |
| Config | OmegaConf + Pydantic | YAML composition + runtime validation |
| Observability | OpenTelemetry + Prometheus | Industry standard, Grafana dashboards |
| Logging | structlog | Structured JSON, context binding |
| Testing | pytest | Markers for unit/integration, timeout support |
| Linting | ruff + mypy | Fast, strict |

## Repo structure

```
src/
  coord/          # Coordinator, registry, health checks
  workers/        # Rollout workers + environments
  trainer/        # PPO, batch collector, checkpointing, eval
  infra/          # Config, logging, metrics, seeding, errors
scripts/          # Run/benchmark/kill utilities
configs/          # Experiment configs (base, churn, cartpole, perf)
tests/            # Unit + integration tests
docker/           # Dockerfile + Prometheus/Grafana stack
docs/             # Runbook
outputs/          # Generated artifacts (gitignored)
```

## Development

```bash
# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/

# Unit tests
pytest -q -m "not integration"

# Integration tests (requires Ray)
pytest -q -m integration

# All tests
pytest -v

# Format
ruff format src/ tests/ scripts/
```

## Design decisions

- **On-policy batch collector, not replay buffer**: PPO is on-policy. Data is discarded after each training step.
- **Policy version tagging**: Every trajectory carries the policy version it was collected under. Stale data is rejected to prevent silent corruption.
- **Async collection with ray.wait**: Workers submit results as they finish. No blocking on the slowest worker.
- **Atomic checkpoints**: Write to `.tmp`, then `os.replace` (atomic on Windows NTFS and POSIX).
- **CartPole-first validation**: Verify PPO on a known-good environment before debugging custom envs.

## Roadmap

- [ ] DDP/FSDP trainer mode for multi-GPU
- [ ] Vectorized environments + async rollouts
- [ ] GRPO-lite algorithm (drop-in via TrainerInterface)
- [ ] Peer-to-peer worker gossip / leader election
- [ ] Better cost model for heterogeneous workers

## License

MIT
