"""End-to-end integration test: full pipeline with coordinator, workers, trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import ray
import torch

from src.coord.coordinator import Coordinator
from src.infra.config import AppConfig
from src.infra.run_context import setup_run_directory
from src.infra.seeding import seed_everything
from src.trainer.batch_collector import RolloutBatchCollector
from src.trainer.checkpoint import CheckpointManager
from src.trainer.eval import Evaluator
from src.trainer.ppo import PPOTrainer, compute_gae
from src.workers.rollout_worker import RolloutWorker

if TYPE_CHECKING:
    from pathlib import Path


def _get_env_dims(config: AppConfig) -> tuple[int, int]:
    from src.workers.envs.gym_env import GymEnvWrapper
    env = GymEnvWrapper("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    return obs_dim, act_dim


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True, num_cpus=4, logging_level=40)
    yield
    ray.shutdown()


@pytest.mark.integration
def test_full_pipeline_cartpole(tmp_output_dir: Path) -> None:
    """Run 3 training steps with 2 workers on CartPole."""
    config = AppConfig(
        seed=42,
        run_id="integration_test",
        env={"name": "cartpole"},
        workers={
            "num_workers": 2,
            "rollout_steps": 64,
            "heterogeneous": False,
            "profiles": [
                {"speed_factor": 1.0, "failure_rate": 0.0, "max_batch": 256},
                {"speed_factor": 1.0, "failure_rate": 0.0, "max_batch": 256},
            ],
        },
        trainer={
            "total_timesteps": 500,
            "batch_size": 64,
            "num_epochs": 2,
            "num_minibatches": 2,
            "lr": 3e-4,
            "normalize_obs": False,
            "normalize_reward": False,
            "eval_interval": 2,
            "eval_episodes": 2,
            "checkpoint_interval": 2,
            "checkpoint_keep": 2,
        },
        coordinator={"heartbeat_timeout_s": 30.0, "persist_state": False},
        churn={"enabled": False},
    )

    run_dir = setup_run_directory(config, base_dir=str(tmp_output_dir))
    seed_everything(config.seed)
    obs_dim, act_dim = _get_env_dims(config)

    trainer = PPOTrainer(obs_dim, act_dim, config.trainer, device=torch.device("cpu"))
    ckpt_mgr = CheckpointManager(run_dir, keep=2)
    evaluator = Evaluator(config, run_dir)

    coordinator = Coordinator.remote(config, str(run_dir))

    workers = []
    for i in range(config.workers.num_workers):
        w = RolloutWorker.remote(
            worker_id=f"worker-{i}",
            config=config,
            speed_factor=1.0,
            failure_rate=0.0,
        )
        ray.get(coordinator.register_worker.remote(f"worker-{i}", w))
        workers.append(w)

    policy_weights = trainer.get_policy_state_dict()
    ray.get(coordinator.update_policy.remote(policy_weights, 0))
    for w in workers:
        w.set_policy_module.remote(trainer.policy_module, 0)

    collector = RolloutBatchCollector(target_batch_size=64, policy_version=0)
    steps_done = 0

    for _iteration in range(3):
        assignments = ray.get(coordinator.get_assignments.remote(collector.fullness))
        refs = {}
        for a in assignments:
            ref = a["actor_handle"].collect_rollout.remote(a["rollout_steps"])
            refs[ref] = a["worker_id"]

        for ref in refs:
            traj = ray.get(ref)
            collector.add(traj)
            ray.get(coordinator.report_rollout_complete.remote(refs[ref]))

        if collector.is_ready():
            batch = compute_gae(collector.trajectories, gamma=0.99, gae_lambda=0.95)
            collector.consume()
            metrics = trainer.update(batch)
            steps_done += 1

            policy_weights = trainer.get_policy_state_dict()
            ray.get(coordinator.update_policy.remote(policy_weights, trainer.global_step))
            collector.set_policy_version(trainer.global_step)
            for w in workers:
                w.set_policy.remote(policy_weights, trainer.global_step)

            assert "policy_loss" in metrics
            assert "value_loss" in metrics

    assert steps_done >= 1

    # Checkpoint and eval
    ckpt_mgr.save(
        step=trainer.global_step,
        trainer_state=trainer.get_checkpoint_state(),
        config_dict=config.model_dump(),
    )
    latest = ckpt_mgr.find_latest()
    assert latest is not None

    eval_result = evaluator.evaluate(trainer.policy_module, trainer.global_step)
    assert "mean_reward" in eval_result
    assert eval_result["mean_reward"] > 0

    status = ray.get(coordinator.get_status.remote())
    assert status["healthy_workers"] == 2


@pytest.mark.integration
def test_checkpoint_resume(tmp_output_dir: Path) -> None:
    """Verify checkpoint save/load/resume produces consistent state."""
    config = AppConfig(
        seed=42,
        run_id="resume_test",
        env={"name": "cartpole"},
        workers={"num_workers": 1, "rollout_steps": 32, "profiles": [{"speed_factor": 1.0}]},
        trainer={
            "total_timesteps": 500,
            "batch_size": 32,
            "num_epochs": 1,
            "num_minibatches": 1,
            "normalize_obs": False,
            "normalize_reward": False,
            "checkpoint_interval": 1,
            "checkpoint_keep": 2,
        },
        coordinator={"persist_state": False},
        churn={"enabled": False},
    )

    run_dir = setup_run_directory(config, base_dir=str(tmp_output_dir))
    seed_everything(config.seed)
    obs_dim, act_dim = _get_env_dims(config)

    trainer = PPOTrainer(obs_dim, act_dim, config.trainer, device=torch.device("cpu"))
    ckpt_mgr = CheckpointManager(run_dir, keep=2)

    # Do a fake update to get some state
    from tests.test_ppo import _make_trajectory
    traj = _make_trajectory(length=32, seed=42)
    batch = compute_gae([traj], gamma=0.99, gae_lambda=0.95)
    trainer.update(batch)

    # Save checkpoint
    ckpt_mgr.save(
        step=trainer.global_step,
        trainer_state=trainer.get_checkpoint_state(),
        config_dict=config.model_dump(),
    )

    # Load into new trainer
    trainer2 = PPOTrainer(obs_dim, act_dim, config.trainer, device=torch.device("cpu"))
    state = ckpt_mgr.load(ckpt_mgr.find_latest())
    trainer2.load_checkpoint_state(state)

    assert trainer2.global_step == trainer.global_step
    for k in trainer.get_policy_state_dict():
        torch.testing.assert_close(
            trainer.get_policy_state_dict()[k],
            trainer2.get_policy_state_dict()[k],
        )
