# Distributed RL Training Platform

> **Built a fault-tolerant distributed RL training system with decentralized rollouts across heterogeneous workers, reproducible experiment tracking, and automated evaluation; supports worker churn, checkpoint recovery, and performance profiling.**

A production-grade distributed ML system that trains policies using reinforcement learning while handling **heterogeneous workers**, **worker churn**, **checkpoint recovery**, and **reproducible experiments**. Bridges "research code" (RL algorithm) to a "systems product" (orchestration, fault tolerance, observability).

**42 tests passing | 0 lint errors | 0 type errors | 3 validated training runs**

## Results

| Scenario | Environment | Workers | Peak Reward | Final Reward | Solved? |
|----------|------------|---------|-------------|--------------|---------|
| Stable | CartPole-v1 | 3 | **500.0** (max) | 492.9 +/- 21.3 | Yes |
| Heterogeneous | JobScheduling | 3 (varied speed) | **0.0** (optimal) | 0.0 +/- 0.0 | Yes |
| Churn | CartPole-v1 | 4 (with kills) | 437.9 | 375.1 +/- 98.2 | Resilient |

![Training performance across all scenarios](experiments/plots/reward_curves.png)

- **CartPole** solved to maximum score (500) by step 40, maintained 490+ through 98 steps
- **Scheduling env** converges from -11.9 to 0.0 (optimal) across heterogeneous workers at 1.0x/0.8x/0.6x speed
- **Churn scenario** survives random worker kills and 2-5% failure rates, reaching 76% of stable performance

Full analysis with diagnostics: [`tests/PERFORMANCE_REPORT.md`](tests/PERFORMANCE_REPORT.md)

<details>
<summary>Training diagnostics (CartPole)</summary>

![CartPole diagnostics: reward, losses, entropy, KL, LR](experiments/plots/cartpole_diagnostics.png)
</details>

<details>
<summary>Fault tolerance under churn</summary>

![Churn analysis: worker count, reward, stable vs churn comparison](experiments/plots/churn_analysis.png)

Worker count grows from 4 to 12 as replacements accumulate. Training completes with no data corruption.
</details>

<details>
<summary>Scheduling environment convergence</summary>

![Scheduling convergence from random to optimal](experiments/plots/scheduling_convergence.png)
</details>

---

## What This Demonstrates

| Competency | Implementation |
|-----------|---------------|
| **Distributed systems** | Ray actor coordination, heartbeat health monitoring, async collection with `ray.wait`, backpressure, worker registry |
| **RL algorithms** | PPO with all CleanRL best practices: GAE, clipped surrogate, value clipping, entropy bonus, advantage normalization, gradient clipping, orthogonal init, linear LR decay |
| **Fault tolerance** | Atomic checkpointing, worker churn detection/replacement, trajectory dedup, WAL for batch recovery, policy version tagging, stale rollout rejection, graceful shutdown |
| **Reproducibility** | Deterministic seeding with per-worker derivation, RNG state checkpointing, config snapshots, run metadata with git SHA |
| **Observability** | OpenTelemetry metrics + Prometheus exporter, Grafana dashboards (8 panels), structured JSON logging (structlog), experiment tracking (local/W&B/MLflow) |
| **Performance** | `torch.compile` support, mixed precision config, PyTorch profiler integration, compute cost tracking per worker |
| **Engineering** | Full type hints + mypy, ruff linting, 42 tests (unit + integration), typed error hierarchy, pre-commit hooks, Docker + Makefile |

---

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

**Data flow:** Coordinator broadcasts policy weights to workers. Workers collect rollouts asynchronously at different speeds and failure rates. Batch collector accumulates on-policy data with dedup and version checking. Trainer runs PPO update when batch is full, discards data (on-policy), and repeats.

## Quickstart

```bash
# Install
python -m venv .venv && .venv\Scripts\activate  # Windows
# source .venv/bin/activate                     # Linux/Mac
pip install -e ".[dev]"

# Run CartPole (fast validation)
python scripts/run_local.py --config configs/cartpole.yaml

# Run scheduling environment (heterogeneous workers)
python scripts/run_local.py --config configs/base.yaml

# Run with worker churn
python scripts/run_local.py --config configs/churn.yaml

# Resume from checkpoint
python scripts/run_local.py --config configs/base.yaml --resume outputs/<run_id>/checkpoints/latest.pt
```

Override any config value from CLI:

```bash
python scripts/run_local.py --config configs/base.yaml trainer.lr=1e-3 seed=123 workers.num_workers=8
```

### Observability stack

```bash
docker compose -f docker/docker-compose.yaml up -d     # Prometheus + Grafana
python scripts/run_local.py --config configs/base.yaml  # metrics on :8000
# Dashboard at http://localhost:3000 (admin/admin)
```

## Fault Tolerance

- Workers fail mid-rollout (simulated via `failure_rate` config) -- detected and replaced automatically
- Coordinator monitors heartbeats, marks timed-out workers dead, spawns replacements
- Trajectories are deduped by `trajectory_id`; stale rollouts (wrong policy version) are rejected
- Checkpoints are atomic (`os.replace`) and include model + optimizer + scheduler + RNG state + normalizer stats
- Batch collector maintains a WAL for crash recovery of partial batches
- Runs resume exactly from checkpoint with full RNG state restoration

## Experiment Reproducibility

Each run captures a complete audit trail:

```
outputs/<run_id>/
  config_resolved.yaml    # full config snapshot (every parameter)
  run_metadata.json       # git SHA, Python/Torch versions, platform, seed
  metrics.jsonl           # per-step training + eval metrics
  checkpoints/            # model + optimizer + RNG state
  eval/                   # per-step evaluation results
  logs/train.jsonl        # structured JSON logs
```

## Observability

| Metric | Type | Description |
|--------|------|-------------|
| `dist_rl.rollout.throughput` | Counter | Trajectories received |
| `dist_rl.trainer.step_duration` | Histogram | Training step wall time |
| `dist_rl.trainer.reward` | Histogram | Evaluation reward |
| `dist_rl.worker.failure_count` | Counter | Worker failures by ID |
| `dist_rl.buffer.depth` | UpDownCounter | Current batch fill level |
| `dist_rl.worker.active_count` | UpDownCounter | Active worker count |

Pre-built Grafana dashboard with 8 panels: reward curve, rollout throughput, trainer step duration, active workers, failure count, buffer depth, KL divergence, checkpoint duration.

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Training | PyTorch 2.x | `torch.compile`, AMP, industry standard |
| Orchestration | Ray | Actor model, built-in fault detection, object store |
| RL | PPO (CleanRL-style) | Well-understood, stable, proven on benchmarks |
| Environment | Gymnasium | Standard RL interface (custom `JobSchedulingEnv` + CartPole) |
| Config | OmegaConf + Pydantic | YAML composition + runtime type validation |
| Metrics | OpenTelemetry + Prometheus | Industry standard, Grafana integration |
| Logging | structlog | Structured JSON with context binding |
| Testing | pytest | Markers for unit/integration, timeout, 42 tests |
| Quality | ruff + mypy | Zero-warning lint + type checking |

## Repo Structure

```
src/
  coord/          # Coordinator actor, worker registry, health monitor
  workers/        # Rollout worker actor, trajectory dataclass, environments
  trainer/        # PPO trainer, batch collector, checkpointing, evaluator, networks
  infra/          # Config, logging, metrics, seeding, errors, profiling, cost tracking
scripts/          # run_local.py, benchmark.py, generate_report.py, kill_workers.py
configs/          # base.yaml, cartpole.yaml, churn.yaml, perf.yaml
tests/            # 42 tests (unit + integration) + PERFORMANCE_REPORT.md
docker/           # Dockerfile + Prometheus/Grafana compose stack
docs/             # Operations runbook
experiments/      # Generated plots (gitignored)
```

## Development

```bash
make lint          # ruff check
make typecheck     # mypy (0 errors)
make test          # unit tests
make test-all      # unit + integration (42 passing)
make format        # ruff format + fix
make docker-up     # start Prometheus + Grafana
```

## Design Decisions

- **On-policy batch collector, not replay buffer.** PPO is on-policy. Data is collected, used once, and discarded. Using a replay buffer would violate PPO's assumptions and cause silent divergence.
- **Policy version tagging.** Every trajectory carries the policy version it was collected under. In a distributed setting with async workers, stale data from an old policy would corrupt gradient estimates. Rejected with a typed `StaleRolloutError`.
- **Async collection with `ray.wait`.** Workers submit results as they finish. No blocking on the slowest worker. Combined with capacity-weighted assignment and backpressure to prevent buffer overflow.
- **Atomic checkpoints.** Write to `.tmp`, then `os.replace()` -- atomic on both Windows NTFS and POSIX. Prevents corrupted checkpoints from partial writes during crashes.
- **CartPole-first validation.** Always verify PPO on a known-good environment (CartPole should reach 500) before debugging a custom environment. This isolates algorithm bugs from environment bugs.
- **Typed error hierarchy.** Seven exception types (`WorkerFailureError`, `RolloutTimeoutError`, `StaleRolloutError`, `CheckpointCorruptionError`, etc.) make error handling explicit and observable.

## Roadmap

- [ ] DDP/FSDP trainer mode for multi-GPU
- [ ] Vectorized environments + async rollouts
- [ ] GRPO-lite algorithm (drop-in via `TrainerInterface` protocol)
- [ ] Peer-to-peer worker gossip / leader election
- [ ] Richer cost model for heterogeneous worker scheduling

## License

MIT
