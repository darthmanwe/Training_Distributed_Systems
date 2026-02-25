"""Benchmark script: runs multiple configs and compares performance."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_local import run_training
from src.infra.config import load_config

CONFIGS = {
    "base": ("configs/base.yaml", ["trainer.total_timesteps=5000"]),
    "cartpole": ("configs/cartpole.yaml", ["trainer.total_timesteps=5000"]),
}


def run_benchmark() -> None:
    results = {}
    for name, (config_path, overrides) in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"{'='*60}\n")

        config = load_config(config_path, overrides=overrides)
        start = time.time()

        try:
            run_training(config)
        except Exception as e:
            print(f"Benchmark {name} failed: {e}")
            continue

        elapsed = time.time() - start

        run_dir = Path("outputs") / str(config.run_id)
        metrics_file = run_dir / "metrics.jsonl"
        final_reward = None
        step_count = 0

        if metrics_file.exists():
            with open(metrics_file) as f:
                for line in f:
                    data = json.loads(line)
                    if "eval" in data:
                        final_reward = data["eval"]["mean_reward"]
                    elif "global_step" in data:
                        step_count = data["global_step"]

        results[name] = {
            "elapsed_s": round(elapsed, 1),
            "steps": step_count,
            "final_reward": final_reward,
            "throughput_steps_per_s": round(step_count / max(elapsed, 1), 1),
        }

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Time(s)':<10} {'Steps':<8} {'Reward':<12} {'Steps/s':<10}")
    print("-" * 55)
    for name, r in results.items():
        reward_str = f"{r['final_reward']:.1f}" if r["final_reward"] else "N/A"
        print(
            f"{name:<15} {r['elapsed_s']:<10} {r['steps']:<8} {reward_str:<12} "
            f"{r['throughput_steps_per_s']:<10}"
        )

    out_path = Path("experiments")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path / 'benchmark_results.json'}")


if __name__ == "__main__":
    run_benchmark()
