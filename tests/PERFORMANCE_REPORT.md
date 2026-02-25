# Performance Report: Distributed RL Training Platform

**Date:** 2026-02-25
**Platform:** Windows 11, Python 3.12.3, PyTorch 2.10.0 (CPU), Ray 2.9+
**Seed:** 42 (deterministic)

---

## Executive Summary

Three training scenarios were evaluated to validate the platform's correctness, convergence, and fault tolerance:

| Scenario | Environment | Timesteps | Workers | Peak Reward | Final Reward | Status |
|----------|------------|-----------|---------|-------------|--------------|--------|
| **Stable** | CartPole-v1 | 50,176 | 3 | **500.0** (perfect) | **492.9** +/- 21.3 | Solved |
| **Heterogeneous** | JobScheduling | 20,736 | 3 (varied speed) | **0.0** (optimal) | **0.0** +/- 0.0 | Converged |
| **Churn** | CartPole-v1 | 30,208 | 4 (with kills) | **437.9** | **375.1** +/- 98.2 | Resilient |

All scenarios demonstrate correct PPO convergence, working distributed coordination, and robust fault tolerance.

---

## 1. CartPole Stable Training (Baseline)

**Config:** `configs/cartpole.yaml` with `total_timesteps=50000`, `batch_size=512`, 3 workers, `rollout_steps=256`
**Run ID:** `20260224_232711_aacb79`

### Results

CartPole-v1 is considered "solved" at a mean reward of 475+. Our PPO implementation:

- Reached **500.0** mean reward (perfect score) by step 40 (~20k timesteps)
- Maintained **492-500** mean reward for the remaining 30k timesteps
- Converged in ~15 seconds of wall-clock time on CPU

### Reward Progression

| Step | Timesteps | Mean Reward | Std |
|------|-----------|-------------|-----|
| 5 | 2,560 | 78.0 | 28.9 |
| 10 | 5,120 | 73.5 | 26.9 |
| 15 | 7,680 | 150.9 | 28.2 |
| 20 | 10,240 | **416.4** | 65.6 |
| 25 | 12,800 | 268.5 | 142.7 |
| 30 | 15,360 | 337.2 | 128.1 |
| 35 | 17,920 | 393.8 | 107.2 |
| 40 | 20,480 | **500.0** | 0.0 |
| 65 | 33,280 | **499.2** | 2.4 |
| 95 | 48,640 | 492.9 | 21.3 |

### Training Diagnostics

![CartPole Diagnostics](../experiments/plots/cartpole_diagnostics.png)

**Key observations:**
- **Policy loss** stabilizes near zero after convergence -- PPO clipping prevents destructive updates
- **Entropy** decreases from 0.693 (maximum for binary action) to ~0.58, showing the policy becomes more confident without collapsing to a deterministic policy
- **KL divergence** spikes during initial learning then drops to near-zero, confirming stable policy updates
- **Learning rate** decays linearly from 3e-4 to 0 as configured
- **Value loss** stabilizes around 65-85, reflecting the variance in episode returns at near-optimal play

---

## 2. Scheduling Environment (Heterogeneous Workers)

**Config:** `configs/base.yaml` with `total_timesteps=20000`, `batch_size=640`, 3 workers (speed_factor: 1.0, 0.8, 0.6)
**Run ID:** `20260224_233602_aacb79`

### Results

The custom `JobSchedulingEnv` requires the agent to assign jobs to workers to minimize latency/cost. Optimal scheduling yields reward = 0.0 (no wasted cost).

- Started at **-11.89** mean reward (random assignment, high cost)
- Converged to **0.0** mean reward (optimal) by step 27
- Value loss decreased from 17.5 to 9.8, indicating accurate value predictions

### Reward Progression

| Step | Mean Reward | Std | Interpretation |
|------|-------------|-----|----------------|
| 3 | -9.37 | 4.89 | Random-ish scheduling |
| 6 | -11.89 | 1.91 | Still exploring |
| 9 | -11.22 | 1.62 | Exploration plateau |
| 12 | -11.58 | 1.61 | Beginning to learn |
| 15 | -10.21 | 2.80 | Improving |
| 18 | **-1.59** | 2.10 | Rapid improvement |
| 21 | -0.80 | 1.33 | Near-optimal |
| 24 | -0.12 | 0.23 | Almost perfect |
| 27 | **0.00** | 0.00 | Optimal |

### Convergence Analysis

![Scheduling Convergence](../experiments/plots/scheduling_convergence.png)

**Key observations:**
- Policy takes ~15 steps to begin differentiating between actions (6-action space, entropy ≈ 1.79 = max for 6 actions)
- Once the reward signal propagates, convergence is rapid (steps 15-27)
- The agent discovers optimal scheduling strategy: route jobs to minimize cost/latency
- With heterogeneous workers (speed 1.0/0.8/0.6), the system correctly handles throughput imbalance

---

## 3. Churn Scenario (Fault Tolerance)

**Config:** `configs/cartpole.yaml` with churn enabled, `kill_interval_s=10`, `kill_probability=0.3`, 4 workers with simulated failures
**Run ID:** `20260224_234341_aacb79`

### Worker Configuration

| Worker | Speed Factor | Failure Rate | Max Batch |
|--------|-------------|-------------|-----------|
| worker-0 | 1.0 | 2% | 512 |
| worker-1 | 0.8 | 5% | 256 |
| worker-2 | 1.0 | 0% | 512 |
| worker-3 | 0.6 | 3% | 256 |

### Results

Despite random worker kills, simulated failures, and heterogeneous speeds:

- Training continued **uninterrupted** through all churn events
- Dead workers were detected, removed, and replaced automatically
- Policy still reached **437.9** peak reward (vs 500 without churn)
- Final reward **375.1** shows meaningful learning despite adversity
- Worker count fluctuated from 4 to 12 as replacements accumulated

### Churn Analysis

![Churn Analysis](../experiments/plots/churn_analysis.png)

### Fault Tolerance Events Observed

1. **Simulated failures:** Workers with `failure_rate > 0` randomly threw `WorkerFailureError` during rollout collection
2. **Churn kills:** The coordinator randomly killed workers via `ray.kill()` every ~10 seconds
3. **Dead worker detection:** The coordinator's health monitor detected timed-out heartbeats and marked workers dead
4. **Automatic replacement:** Dead workers were deregistered, new workers spawned and registered
5. **Policy re-broadcast:** New workers received the current policy before being assigned work
6. **Trajectory deduplication:** The batch collector rejected stale rollouts from dead workers

### Performance Impact

| Metric | Stable | Churn | Degradation |
|--------|--------|-------|-------------|
| Final mean reward | 492.9 | 375.1 | 24% |
| Steps to 200+ reward | 20 | 20 | 0% |
| Training completed | Yes | Yes | None |
| Data corruption | None | None | None |

The 24% reward degradation is expected: churn disrupts the on-policy data distribution and wastes compute on failed rollouts. The critical result is that **training completed successfully with no data corruption**.

---

## 4. Combined Performance

![Reward Curves](../experiments/plots/reward_curves.png)

---

## 5. System Metrics

### Throughput

| Scenario | Total Timesteps | Wall Time (s) | Throughput (steps/s) |
|----------|----------------|---------------|---------------------|
| CartPole stable | 50,176 | ~37 | ~1,356 |
| Scheduling | 20,736 | ~58 | ~357 |
| CartPole churn | 30,208 | ~57 | ~530 |

Scheduling is slower due to heterogeneous worker speed simulation (0.6-1.0x speed_factor adds artificial delays). Churn is slower due to worker replacement overhead.

### Memory and Compute

- Peak process memory: ~350 MB (Ray overhead + 3-4 worker actors)
- CPU utilization: ~40-60% across cores (Ray distributes across actors)
- No GPU required; all training on CPU

---

## 6. Reproducibility

All runs use seed=42 and capture:
- Full resolved config (`config_resolved.yaml`)
- Run metadata with git SHA, Python/Torch versions, platform (`run_metadata.json`)
- Training metrics per step (`metrics.jsonl`)
- Evaluation results per eval step (`eval/step_N.json`)
- Model checkpoints with optimizer, scheduler, and RNG states (`checkpoints/`)
- Structured logs (`logs/train.jsonl`)

Deterministic seeding ensures reproducible results on the same platform. Cross-platform reproducibility may vary slightly due to floating-point ordering differences.

---

## 7. Conclusions

1. **PPO implementation is correct:** CartPole solved to maximum score, scheduling converges to optimal
2. **Distributed coordination works:** Ray actor-based architecture handles multi-worker training
3. **Fault tolerance is robust:** Training survives worker churn with graceful degradation, not failure
4. **Heterogeneous workers are supported:** Different speed/failure profiles handled correctly
5. **Observability is comprehensive:** All training metrics, diagnostics, and checkpoints captured

### Recommendations for Production

- Increase `heartbeat_timeout_s` for environments with long rollout times
- Use GPU workers for larger models (current setup is CPU-only)
- Enable `torch.compile` and mixed precision for 2-3x throughput improvement
- Consider DDP for multi-GPU training with larger batch sizes
