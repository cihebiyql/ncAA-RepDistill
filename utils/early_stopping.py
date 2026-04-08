"""Early stopping helpers for training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepEarlyStopping:
    """Step-based early stopping tracker."""

    patience_steps: int
    mode: str = "min"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience_steps <= 0:
            raise ValueError("patience_steps must be positive")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.best_metric = None
        self.best_step = 0
        self.triggered = False

    def update(self, metric_value: float, step: int) -> bool:
        """Update tracker and return True if early stopping should trigger."""

        if metric_value is None:
            return False

        improved = False
        if self.best_metric is None:
            improved = True
        else:
            if self.mode == "min":
                improved = metric_value < self.best_metric - self.min_delta
            else:
                improved = metric_value > self.best_metric + self.min_delta

        if improved:
            self.best_metric = metric_value
            self.best_step = step
            return False

        if step - self.best_step >= self.patience_steps:
            self.triggered = True
            return True

        return False


__all__ = ["StepEarlyStopping"]

