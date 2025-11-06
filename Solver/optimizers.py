"""Optimizer implementations for custom solver workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .cost_functions import cost_sensitive_nll, cost_sensitive_nll_gradient


@dataclass
class GradientDescentConfig:
    max_iter: int = 500
    learning_rate: float = 0.1
    tolerance: float = 1e-5
    momentum: float = 0.0
    verbose: bool = False
    track_history: bool = False


def gradient_descent_cost_sensitive(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    config: GradientDescentConfig,
    weights_init: np.ndarray | None = None,
    bias_init: float = 0.0,
) -> Dict[str, object]:
    """Perform gradient descent on the cost-sensitive NLL objective."""

    n_features = X.shape[1]
    weights = np.zeros(n_features, dtype=float) if weights_init is None else weights_init.astype(float)
    bias = float(bias_init)

    velocity_w = np.zeros_like(weights)
    velocity_b = 0.0

    history: Dict[str, List[float]] = {"loss": []} if config.track_history else {}

    prev_loss = np.inf
    for iteration in range(1, config.max_iter + 1):
        loss = cost_sensitive_nll(weights, bias, X, y, sample_weight)
        grad_w, grad_b = cost_sensitive_nll_gradient(weights, bias, X, y, sample_weight)

        velocity_w = config.momentum * velocity_w + grad_w
        velocity_b = config.momentum * velocity_b + grad_b

        weights -= config.learning_rate * velocity_w
        bias -= config.learning_rate * velocity_b

        if config.track_history:
            history.setdefault("loss", []).append(loss)

        if config.verbose and iteration % 50 == 0:
            print(f"[Solver] Iter {iteration:04d} | Loss={loss:.6f}")

        if abs(prev_loss - loss) < config.tolerance:
            break
        prev_loss = loss

    return {
        "weights": weights,
        "bias": bias,
        "iterations": iteration,
        "history": history,
        "final_loss": loss,
    }


__all__ = ["GradientDescentConfig", "gradient_descent_cost_sensitive"]

