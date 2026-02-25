"""Trainer interface protocol: allows PPO and GRPO-lite to be swapped."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.workers.trajectory import TrajectoryBatch


class TrainerInterface(Protocol):
    """Protocol that any RL trainer must implement."""

    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        """Run one training iteration on a batch. Returns metrics dict."""
        ...

    def get_policy_state_dict(self) -> dict[str, Any]:
        """Return serializable policy weights."""
        ...

    def load_policy_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load policy weights."""
        ...

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return full trainer state for checkpointing."""
        ...

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore full trainer state from checkpoint."""
        ...

    @property
    def policy_module(self) -> Any:
        """Return the underlying nn.Module for weight broadcasting."""
        ...

    @property
    def global_step(self) -> int:
        """Current training step count."""
        ...
