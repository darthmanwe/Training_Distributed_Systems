"""Main entrypoint: runs the full distributed RL training pipeline locally."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import ray

from src.coord.coordinator import Coordinator
from src.infra.config import AppConfig, load_config
from src.infra.run_context import setup_run_directory
from src.infra.seeding import seed_everything
from src.trainer.batch_collector import RolloutBatchCollector
from src.trainer.checkpoint import CheckpointManager
from src.trainer.eval import Evaluator
from src.trainer.ppo import PPOTrainer, compute_gae
from src.workers.rollout_worker import RolloutWorker

logger = logging.getLogger(__name__)

SHUTDOWN_REQUESTED = False


def _setup_logging(level: str, run_dir: Path) -> None:
    log_file = run_dir / "logs" / "train.jsonl"

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            d = {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "msg": record.getMessage(),
                "logger": record.name,
            }
            if hasattr(record, "worker_id"):
                d["worker_id"] = record.worker_id  # type: ignore[attr-defined]
            return json.dumps(d)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console)

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)


def _signal_handler(signum: int, frame: Any) -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    logger.info("Shutdown signal received, finishing current step...")


def _get_env_dims(config: AppConfig) -> tuple[int, int]:
    """Probe the environment to get observation and action dimensions."""
    from src.workers.envs.gym_env import GymEnvWrapper
    from src.workers.envs.scheduling_env import JobSchedulingEnv

    if config.env.name == "cartpole":
        env = GymEnvWrapper("CartPole-v1")
    else:
        env = JobSchedulingEnv(
            num_env_workers=config.env.num_env_workers,
            num_jobs=config.env.num_jobs,
            max_queue_len=config.env.max_queue_len,
            max_steps_per_episode=config.env.max_steps_per_episode,
        )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # type: ignore[union-attr]
    return obs_dim, act_dim


def _spawn_workers(
    config: AppConfig, coordinator: Any
) -> list[Any]:
    """Spawn rollout worker actors and register them with the coordinator."""
    workers = []
    profiles = config.workers.profiles
    for i in range(config.workers.num_workers):
        profile = profiles[i] if i < len(profiles) else profiles[-1]
        wid = f"worker-{i}"
        worker = RolloutWorker.remote(
            worker_id=wid,
            config=config,
            speed_factor=profile.speed_factor,
            failure_rate=profile.failure_rate,
            max_batch=profile.max_batch,
        )
        ray.get(
            coordinator.register_worker.remote(
                worker_id=wid,
                actor_handle=worker,
                speed_factor=profile.speed_factor,
                failure_rate=profile.failure_rate,
                max_batch=profile.max_batch,
            )
        )
        workers.append(worker)
        logger.info(f"Spawned {wid} (speed={profile.speed_factor}, fail={profile.failure_rate})")
    return workers


def _replace_dead_workers(
    config: AppConfig, coordinator: Any, alive_workers: list[Any]
) -> list[Any]:
    """Detect dead workers and spawn replacements."""
    dead = ray.get(coordinator.remove_dead_workers.remote())
    if not dead:
        return alive_workers

    alive_handles = set()
    for w in alive_workers:
        try:
            wid = ray.get(w.ping.remote(), timeout=2)
            alive_handles.add(wid)
        except Exception:
            pass

    new_workers = [w for w in alive_workers if _worker_alive(w)]

    profiles = config.workers.profiles
    for _i, wid in enumerate(dead):
        idx = int(wid.split("-")[-1])
        profile = profiles[idx] if idx < len(profiles) else profiles[-1]
        new_wid = f"worker-{idx}"
        worker = RolloutWorker.remote(
            worker_id=new_wid,
            config=config,
            speed_factor=profile.speed_factor,
            failure_rate=profile.failure_rate,
            max_batch=profile.max_batch,
        )
        ray.get(
            coordinator.register_worker.remote(
                worker_id=new_wid,
                actor_handle=worker,
                speed_factor=profile.speed_factor,
                failure_rate=profile.failure_rate,
                max_batch=profile.max_batch,
            )
        )
        new_workers.append(worker)
        logger.info(f"Replaced dead worker {wid} -> {new_wid}")

    return new_workers


def _worker_alive(worker: Any) -> bool:
    try:
        ray.get(worker.ping.remote(), timeout=2)
        return True
    except Exception:
        return False


def _churn_kill(config: AppConfig, workers: list[Any], coordinator: Any) -> list[Any]:
    """Randomly kill a worker to simulate churn."""
    import random

    if not config.churn.enabled or not workers:
        return workers
    if random.random() > config.churn.kill_probability:
        return workers

    victim = random.choice(workers)
    try:
        wid = ray.get(victim.ping.remote(), timeout=2)
        logger.warning(f"CHURN: killing {wid}")
        ray.kill(victim)
    except Exception:
        pass

    return [w for w in workers if w is not victim]


def run_training(config: AppConfig, resume_path: str | None = None) -> None:
    """Main training loop."""
    seed_everything(config.seed)
    run_dir = setup_run_directory(config)
    _setup_logging(config.obs.log_level, run_dir)

    logger.info(f"Starting run {config.run_id} (seed={config.seed})")
    logger.info(f"Output directory: {run_dir}")

    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    obs_dim, act_dim = _get_env_dims(config)
    logger.info(f"Environment: {config.env.name} (obs_dim={obs_dim}, act_dim={act_dim})")

    # Initialize trainer
    trainer = PPOTrainer(obs_dim, act_dim, config.trainer)
    ckpt_mgr = CheckpointManager(run_dir, keep=config.trainer.checkpoint_keep)
    evaluator = Evaluator(config, run_dir)

    # Resume from checkpoint
    if resume_path:
        state = ckpt_mgr.load(Path(resume_path))
        trainer.load_checkpoint_state(state)
        logger.info(f"Resumed from step {trainer.global_step}")

    # Spawn coordinator and workers
    coordinator = Coordinator.remote(config, str(run_dir))
    workers = _spawn_workers(config, coordinator)

    # Broadcast initial policy to all workers
    policy_weights = trainer.get_policy_state_dict()
    policy_version = trainer.global_step
    ray.get(coordinator.update_policy.remote(policy_weights, policy_version))

    for w in workers:
        try:
            w.set_policy_module.remote(trainer.policy_module, policy_version)
        except Exception:
            w.set_policy.remote(policy_weights, policy_version)

    batch_collector = RolloutBatchCollector(
        target_batch_size=config.trainer.batch_size,
        policy_version=policy_version,
        wal_path=run_dir / "logs" / "buffer_wal.jsonl",
    )

    total_timesteps = 0
    last_churn_time = time.time()
    metrics_file = run_dir / "metrics.jsonl"

    # Register cleanup
    def _cleanup() -> None:
        logger.info("Cleaning up...")
        with contextlib.suppress(Exception):
            ray.get(coordinator.shutdown.remote(), timeout=5)
        ray.shutdown()

    atexit.register(_cleanup)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    logger.info("Starting training loop...")

    while total_timesteps < config.trainer.total_timesteps and not SHUTDOWN_REQUESTED:
        # Churn simulation
        if config.churn.enabled and (time.time() - last_churn_time) > config.churn.kill_interval_s:
            workers = _churn_kill(config, workers, coordinator)
            workers = _replace_dead_workers(config, workers, workers)
            last_churn_time = time.time()

        # Get rollout assignments (with backpressure)
        assignments = ray.get(
            coordinator.get_assignments.remote(batch_collector.fullness)
        )

        # Dispatch rollouts asynchronously
        pending_refs = {}
        for assign in assignments:
            worker_handle = assign["actor_handle"]
            steps = assign["rollout_steps"]
            try:
                ref = worker_handle.collect_rollout.remote(steps)
                pending_refs[ref] = assign["worker_id"]
            except Exception as e:
                logger.warning(f"Failed to dispatch to {assign['worker_id']}: {e}")

        if not pending_refs:
            time.sleep(0.1)
            workers = _replace_dead_workers(config, workers, workers)
            # Re-broadcast weights to any new workers
            for w in workers:
                with contextlib.suppress(Exception):
                    w.set_policy.remote(policy_weights, policy_version)
            continue

        # Collect results as they arrive (async with ray.wait)
        remaining = list(pending_refs.keys())
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=30.0)
            for ref in ready:
                wid = pending_refs.get(ref, "unknown")
                try:
                    trajectory = ray.get(ref)
                    batch_collector.add(trajectory)
                    ray.get(coordinator.report_rollout_complete.remote(wid))
                except Exception as e:
                    logger.warning(f"Rollout from {wid} failed: {e}")
                    with contextlib.suppress(Exception):
                        ray.get(coordinator.report_rollout_complete.remote(wid))

            if batch_collector.is_ready():
                break

        # If not enough data yet, continue collecting
        if not batch_collector.is_ready():
            continue

        # Compute GAE across all trajectories
        batch = compute_gae(
            batch_collector.trajectories,
            gamma=config.trainer.gamma,
            gae_lambda=config.trainer.gae_lambda,
        )

        # Normalize rewards if configured
        if trainer._reward_normalizer is not None:
            batch = type(batch)(
                observations=batch.observations,
                actions=batch.actions,
                rewards=trainer._reward_normalizer.normalize(batch.rewards, batch.dones),
                dones=batch.dones,
                log_probs=batch.log_probs,
                values=batch.values,
                advantages=batch.advantages,
                returns=batch.returns,
            )

        # Consume batch from collector (clears it for on-policy)
        batch_collector.consume()

        # PPO update
        metrics = trainer.update(batch)
        total_timesteps += batch.size
        policy_version = trainer.global_step

        # Broadcast updated weights
        policy_weights = trainer.get_policy_state_dict()
        ray.get(coordinator.update_policy.remote(policy_weights, policy_version))
        batch_collector.set_policy_version(policy_version)

        for w in workers:
            with contextlib.suppress(Exception):
                w.set_policy.remote(policy_weights, policy_version)

        # Log metrics
        metrics["total_timesteps"] = total_timesteps
        metrics["num_workers"] = len(workers)
        metrics["dedup_hits"] = batch_collector.dedup_hits

        logger.info(
            f"Step {trainer.global_step}: "
            f"reward_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"kl={metrics['approx_kl']:.4f}, "
            f"lr={metrics['learning_rate']:.2e}, "
            f"timesteps={total_timesteps}"
        )

        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Periodic eval
        if trainer.global_step % config.trainer.eval_interval == 0:
            eval_result = evaluator.evaluate(trainer.policy_module, trainer.global_step)
            logger.info(
                f"Eval step {trainer.global_step}: "
                f"mean_reward={eval_result['mean_reward']:.2f} "
                f"(+/- {eval_result['std_reward']:.2f})"
            )
            with open(metrics_file, "a") as f:
                f.write(json.dumps({"eval": eval_result}) + "\n")

        # Periodic checkpoint (only after successful update -- idempotent)
        if trainer.global_step % config.trainer.checkpoint_interval == 0:
            ckpt_mgr.save(
                step=trainer.global_step,
                trainer_state=trainer.get_checkpoint_state(),
                config_dict=config.model_dump(),
            )

    # Final checkpoint + eval
    if trainer.global_step > 0:
        logger.info("Training complete. Final checkpoint and eval...")
        ckpt_mgr.save(
            step=trainer.global_step,
            trainer_state=trainer.get_checkpoint_state(),
            config_dict=config.model_dump(),
        )
        eval_result = evaluator.evaluate(trainer.policy_module, trainer.global_step)
        logger.info(
            f"Final eval: mean_reward={eval_result['mean_reward']:.2f} "
            f"(+/- {eval_result['std_reward']:.2f})"
        )

    logger.info(f"Run {config.run_id} finished. Total timesteps: {total_timesteps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed RL Training")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "overrides", nargs="*", help="Config overrides in dotlist format (e.g. trainer.lr=1e-3)"
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides if args.overrides else None)
    run_training(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
