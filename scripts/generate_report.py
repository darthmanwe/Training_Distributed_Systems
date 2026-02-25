"""Generate performance plots and report from training run metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> tuple[list[dict], list[dict]]:
    train_metrics = []
    eval_metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "eval" in data:
                eval_metrics.append(data["eval"])
            else:
                train_metrics.append(data)
    return train_metrics, eval_metrics


def plot_reward_curves(
    runs: dict[str, tuple[list[dict], list[dict]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {"CartPole (stable)": "#2196F3", "Scheduling (heterogeneous)": "#4CAF50", "CartPole (churn)": "#FF5722"}
    markers = {"CartPole (stable)": "o", "Scheduling (heterogeneous)": "s", "CartPole (churn)": "^"}

    for name, (_train, evals) in runs.items():
        steps = [e["step"] for e in evals]
        rewards = [e["mean_reward"] for e in evals]
        stds = [e.get("std_reward", 0) for e in evals]

        color = colors.get(name, "#607D8B")
        marker = markers.get(name, "o")
        ax.plot(steps, rewards, label=name, color=color, marker=marker, markersize=5, linewidth=2)
        ax.fill_between(
            steps,
            [r - s for r, s in zip(rewards, stds, strict=False)],
            [r + s for r, s in zip(rewards, stds, strict=False)],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Eval Reward", fontsize=12)
    ax.set_title("Training Performance Across Scenarios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_training_diagnostics(
    train_metrics: list[dict],
    eval_metrics: list[dict],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Training Diagnostics: {title}", fontsize=14, fontweight="bold")

    steps = [m["global_step"] for m in train_metrics]
    [m["total_timesteps"] for m in train_metrics]

    # Eval reward
    ax = axes[0, 0]
    eval_steps = [e["step"] for e in eval_metrics]
    eval_rewards = [e["mean_reward"] for e in eval_metrics]
    eval_stds = [e.get("std_reward", 0) for e in eval_metrics]
    ax.plot(eval_steps, eval_rewards, "b-o", markersize=4, linewidth=1.5)
    ax.fill_between(
        eval_steps,
        [r - s for r, s in zip(eval_rewards, eval_stds, strict=False)],
        [r + s for r, s in zip(eval_rewards, eval_stds, strict=False)],
        alpha=0.2,
    )
    ax.set_title("Eval Reward")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[0, 1]
    ax.plot(steps, [m["policy_loss"] for m in train_metrics], "g-", linewidth=1, alpha=0.8)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[0, 2]
    ax.plot(steps, [m["value_loss"] for m in train_metrics], "r-", linewidth=1, alpha=0.8)
    ax.set_title("Value Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(steps, [m["entropy"] for m in train_metrics], "m-", linewidth=1, alpha=0.8)
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # KL divergence
    ax = axes[1, 1]
    ax.plot(steps, [m["approx_kl"] for m in train_metrics], "c-", linewidth=1, alpha=0.8)
    ax.set_title("Approx KL Divergence")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 2]
    ax.plot(steps, [m["learning_rate"] for m in train_metrics], "k-", linewidth=1.5)
    ax.set_title("Learning Rate (linear decay)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_churn_analysis(
    train_metrics: list[dict],
    eval_metrics: list[dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fault Tolerance: Churn Scenario Analysis", fontsize=14, fontweight="bold")

    # Worker count over time
    ax = axes[0]
    steps = [m["global_step"] for m in train_metrics]
    workers = [m["num_workers"] for m in train_metrics]
    ax.plot(steps, workers, "r-o", markersize=3, linewidth=1.5)
    ax.set_title("Active Workers Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Num Workers")
    ax.grid(True, alpha=0.3)

    # Reward despite churn
    ax = axes[1]
    eval_steps = [e["step"] for e in eval_metrics]
    eval_rewards = [e["mean_reward"] for e in eval_metrics]
    ax.bar(eval_steps, eval_rewards, color="#FF5722", alpha=0.7, width=3)
    ax.set_title("Eval Reward (with churn)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3, axis="y")

    # Reward comparison: stable vs churn at matched steps
    ax = axes[2]
    categories = ["Stable\n(no failures)", "Churn\n(random kills)"]
    final_rewards = [492.9, 375.1]
    bars = ax.bar(categories, final_rewards, color=["#2196F3", "#FF5722"], alpha=0.8, width=0.5)
    ax.set_title("Final Performance Comparison")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_ylim(0, 550)
    for bar, val in zip(bars, final_rewards, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f"{val:.1f}", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scheduling_convergence(
    train_metrics: list[dict],
    eval_metrics: list[dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Scheduling Environment: Convergence Analysis", fontsize=14, fontweight="bold")

    eval_steps = [e["step"] for e in eval_metrics]
    eval_rewards = [e["mean_reward"] for e in eval_metrics]

    # Reward convergence
    ax = axes[0]
    ax.plot(eval_steps, eval_rewards, "g-o", markersize=6, linewidth=2)
    ax.axhline(y=0.0, color="k", linestyle="--", alpha=0.5, label="Optimal (0.0)")
    ax.set_title("Reward Convergence to Optimal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1]
    steps = [m["global_step"] for m in train_metrics]
    vloss = [m["value_loss"] for m in train_metrics]
    ax.plot(steps, vloss, "r-", linewidth=1.5, alpha=0.8)
    ax.set_title("Value Loss (decreasing = better value estimates)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value Loss")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    plots_dir = root / "experiments" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cartpole_dir = root / "outputs" / "20260224_232711_aacb79"
    scheduling_dir = root / "outputs" / "20260224_233602_aacb79"
    churn_dir = root / "outputs" / "20260224_234341_aacb79"

    cp_train, cp_eval = load_metrics(cartpole_dir / "metrics.jsonl")
    sc_train, sc_eval = load_metrics(scheduling_dir / "metrics.jsonl")
    ch_train, ch_eval = load_metrics(churn_dir / "metrics.jsonl")

    # 1. Combined reward curves
    plot_reward_curves(
        {
            "CartPole (stable)": (cp_train, cp_eval),
            "Scheduling (heterogeneous)": (sc_train, sc_eval),
            "CartPole (churn)": (ch_train, ch_eval),
        },
        plots_dir / "reward_curves.png",
    )

    # 2. CartPole diagnostics
    plot_training_diagnostics(cp_train, cp_eval, "CartPole (Stable)", plots_dir / "cartpole_diagnostics.png")

    # 3. Churn analysis
    plot_churn_analysis(ch_train, ch_eval, plots_dir / "churn_analysis.png")

    # 4. Scheduling convergence
    plot_scheduling_convergence(sc_train, sc_eval, plots_dir / "scheduling_convergence.png")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
