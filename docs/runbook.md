# Operations Runbook

## Training is stuck (no progress)

**Symptoms**: No new training steps logged, metrics flat.

1. **Check buffer depth**: Look at `dist_rl.buffer.depth` metric or logs for `batch_collector.fullness`. If 0 or very low, workers aren't producing.
2. **Check worker health**: Run `ray status` or look at coordinator status logs. Are workers alive? Check heartbeat latency.
3. **Check coordinator logs**: Is it dispatching assignments? Look for `backpressure.engaged` -- if fullness > 0.8, the coordinator throttles.
4. **Check for stale rollout rejections**: Look for `StaleRolloutError` in logs. Workers may be too slow. Increase `staleness_window` or reduce batch size.
5. **Emergency**: Kill and restart. If resuming from checkpoint, use `--resume outputs/<run_id>/checkpoints/latest.pt`.

## Reward is not improving

**Symptoms**: Mean eval reward is flat or declining.

1. **Check entropy**: If entropy is near 0, the policy collapsed to a deterministic (bad) action. Increase `ent_coef` (e.g., 0.01 -> 0.05).
2. **Check KL divergence**: If `approx_kl` > 0.05, updates are too aggressive. Reduce `lr` or set `target_kl=0.02` for early stopping.
3. **Check clip fraction**: If > 0.3, the policy is changing too fast per step. Reduce `clip_eps` or `num_epochs`.
4. **Check explained variance**: If near 0 or negative, the value function isn't fitting well. Increase `vf_coef`, check network size.
5. **Check learning rate**: Has it decayed to 0? The linear schedule goes to 0 at `total_timesteps`. Ensure you haven't exceeded it.
6. **Validate on CartPole**: If CartPole doesn't learn either, the PPO implementation has a bug. If it does, the issue is environment-specific.

## Checkpoint recovery

1. **Find latest checkpoint**:
   ```
   ls outputs/<run_id>/checkpoints/
   ```

2. **Resume training**:
   ```
   python scripts/run_local.py --config configs/base.yaml --resume outputs/<run_id>/checkpoints/latest.pt
   ```

3. **Verify resume**: Check that the first log line shows "Resumed from step N" with the expected step. Metrics should continue from where they left off.

4. **If checkpoint is corrupt**: The `.tmp` file shouldn't exist (atomic writes). If it does, the previous save was interrupted. Use the second-most-recent `step_*.pt` file instead.

## Worker crash loop

**Symptoms**: Workers keep dying and being replaced.

1. **Check failure_rate config**: If using `churn.yaml`, high `failure_rate` values (>0.2) cause frequent deaths by design.
2. **Check worker logs**: OOM? Increase `max_batch` limit or reduce `rollout_steps`.
3. **Check resource limits**: `ray status` shows resource allocation. Are workers fighting for resources?
4. **Reduce workers**: Try `workers.num_workers=2` to see if the problem is resource contention.

## Stale rollouts being rejected

**Symptoms**: Lots of `StaleRolloutError` in logs, training is slow.

1. Workers are collecting too slowly relative to trainer. Options:
   - Increase `staleness_window` (allow older policy versions)
   - Reduce `trainer.batch_size` (trainer waits for less data)
   - Increase `workers.rollout_steps` (workers send larger chunks less often)
   - Remove slowest workers (`speed_factor < 0.2` is very slow)

## Memory growing unbounded

**Symptoms**: Process memory keeps increasing.

1. **Check batch collector**: Is `fullness` always near 1.0? The trainer may not be consuming batches.
2. **Check for trainer errors**: If the trainer crashes during `update()`, batches accumulate.
3. **Reduce batch size**: Smaller batches use less memory.
4. **Check GPU memory**: `torch.cuda.max_memory_allocated()` is tracked in cost metrics.

## Prometheus/Grafana not showing data

1. **Check metrics endpoint**: `curl http://localhost:8000/metrics` should return Prometheus-format data.
2. **Check Prometheus config**: `docker/prometheus/prometheus.yml` should target `host.docker.internal:8000`.
3. **Check Grafana datasource**: Settings > Data Sources > Prometheus should point to `http://prometheus:9090`.
4. **Restart stack**: `docker compose -f docker/docker-compose.yaml down && docker compose -f docker/docker-compose.yaml up -d`
