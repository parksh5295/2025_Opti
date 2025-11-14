"""Optimizer implementations for custom solver workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .cost_functions import (
    cost_sensitive_nll,
    cost_sensitive_nll_full,
    cost_sensitive_nll_gradient,
)


@dataclass
class GradientDescentConfig:
    max_iter: int = 500
    learning_rate: float = 0.1
    tolerance: float = 1e-5
    momentum: float = 0.0
    verbose: bool = False
    track_history: bool = False
    method: str = "gd"  # gd, momentum, nesterov, adam
    line_search: bool = False
    line_search_alpha: float = 0.3
    line_search_beta: float = 0.8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_second_order: bool = False
    second_order_method: str = "none"  # "none" or "bfgs"
    track_snapshots: bool = False
    snapshot_interval: int = 5


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
    if weights_init is None:
        # Xavier/Glorot initialization for better convergence
        # Scale by sqrt(1/n_features) for sigmoid activation
        limit = np.sqrt(1.0 / n_features)
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        weights = rng.uniform(-limit, limit, size=n_features).astype(float)
    else:
        weights = weights_init.astype(float)
    if bias_init == 0.0:
        rng = np.random.RandomState(42)
        bias = float(rng.uniform(-0.01, 0.01))
    else:
        bias = float(bias_init)

    velocity_w = np.zeros_like(weights)
    velocity_b = 0.0

    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = 0.0
    v_b = 0.0

    history: Dict[str, List[float]] = {"loss": []} if config.track_history else {}
    snapshots: List[Dict[str, object]] = [] if config.track_snapshots else []

    prev_loss = np.inf
    beta1 = config.adam_beta1
    beta2 = config.adam_beta2
    epsilon = config.adam_epsilon

    if config.track_snapshots:
        snapshots.append({"iteration": 0, "weights": weights.copy(), "bias": bias})

    for iteration in range(1, config.max_iter + 1):
        loss = cost_sensitive_nll(weights, bias, X, y, sample_weight)
        grad_w, grad_b = cost_sensitive_nll_gradient(weights, bias, X, y, sample_weight)

        if config.method == "adam":
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

            m_w_hat = m_w / (1 - beta1 ** iteration)
            m_b_hat = m_b / (1 - beta1 ** iteration)
            v_w_hat = v_w / (1 - beta2 ** iteration)
            v_b_hat = v_b / (1 - beta2 ** iteration)

            direction_w = -m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            direction_b = -m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        else:
            if config.method == "nesterov":
                lookahead_w = weights - config.momentum * velocity_w
                lookahead_b = bias - config.momentum * velocity_b
                grad_w, grad_b = cost_sensitive_nll_gradient(lookahead_w, lookahead_b, X, y, sample_weight)

            velocity_w = config.momentum * velocity_w + grad_w
            velocity_b = config.momentum * velocity_b + grad_b
            direction_w = -velocity_w
            direction_b = -velocity_b

            if config.method == "gd":
                direction_w = -grad_w
                direction_b = -grad_b

        step = config.learning_rate
        if config.line_search:
            step = _backtracking_line_search(
                weights,
                bias,
                direction_w,
                direction_b,
                grad_w,
                grad_b,
                loss,
                X,
                y,
                sample_weight,
                step,
                config.line_search_alpha,
                config.line_search_beta,
            )

        weights += step * direction_w
        bias += step * direction_b

        if config.track_history:
            history.setdefault("loss", []).append(loss)

        if config.track_snapshots and (
            iteration % config.snapshot_interval == 0 or iteration == config.max_iter
        ):
            snapshots.append({"iteration": iteration, "weights": weights.copy(), "bias": bias})

        if config.verbose and iteration % 50 == 0:
            print(f"[Solver] Iter {iteration:04d} | Loss={loss:.6f}")

        if abs(prev_loss - loss) < config.tolerance:
            if config.track_snapshots and snapshots and snapshots[-1]["iteration"] != iteration:
                snapshots.append({"iteration": iteration, "weights": weights.copy(), "bias": bias})
            break
        prev_loss = loss

    return {
        "weights": weights,
        "bias": bias,
        "iterations": iteration,
        "history": history,
        "final_loss": loss,
        "snapshots": snapshots,
    }


def bfgs_cost_sensitive(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    config: GradientDescentConfig,
    theta_init: np.ndarray | None = None,
) -> Dict[str, object]:
    """Quasi-Newton BFGS optimisation for the cost-sensitive NLL."""

    n_samples, n_features = X.shape
    X_aug = np.hstack([X, np.ones((n_samples, 1))])

    if theta_init is None:
        # Xavier initialization
        limit = np.sqrt(1.0 / n_features)
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        theta = rng.uniform(-limit, limit, size=n_features + 1).astype(float)
    else:
        theta = theta_init.astype(float)
    H = np.eye(n_features + 1)

    history: Dict[str, List[float]] = {"loss": []} if config.track_history else {}
    snapshots: List[Dict[str, object]] = [] if config.track_snapshots else []

    if config.track_snapshots:
        snapshots.append({"iteration": 0, "weights": theta[:-1].copy(), "bias": float(theta[-1])})

    for iteration in range(1, config.max_iter + 1):
        loss, grad = cost_sensitive_nll_full(theta, X_aug, y, sample_weight)

        direction = -H @ grad

        grad_dot_dir = float(grad @ direction)
        step = config.learning_rate
        step = _backtracking_line_search_full(
            theta,
            direction,
            grad_dot_dir,
            loss,
            X_aug,
            y,
            sample_weight,
            step,
            config.line_search_alpha,
            config.line_search_beta,
        )

        theta_new = theta + step * direction
        new_loss, grad_new = cost_sensitive_nll_full(theta_new, X_aug, y, sample_weight)

        if config.track_history:
            history.setdefault("loss", []).append(loss)

        s = theta_new - theta
        yk = grad_new - grad
        ys = float(yk @ s)

        if ys <= 1e-12:
            # Skip update to maintain positive definiteness
            theta = theta_new
            continue

        rho = 1.0 / ys
        I = np.eye(n_features + 1)
        H = (I - rho * np.outer(s, yk)) @ H @ (I - rho * np.outer(yk, s)) + rho * np.outer(s, s)

        if config.verbose and iteration % 50 == 0:
            print(f"[Solver-BFGS] Iter {iteration:04d} | Loss={loss:.6f}")

        if config.track_snapshots and (
            iteration % config.snapshot_interval == 0 or iteration == config.max_iter
        ):
            snapshots.append({"iteration": iteration, "weights": theta_new[:-1].copy(), "bias": float(theta_new[-1])})

        if np.linalg.norm(s) < config.tolerance:
            if config.track_snapshots and snapshots and snapshots[-1]["iteration"] != iteration:
                snapshots.append({"iteration": iteration, "weights": theta_new[:-1].copy(), "bias": float(theta_new[-1])})
            theta = theta_new
            break

        theta = theta_new

    weights = theta[:-1]
    bias = float(theta[-1])

    return {
        "weights": weights,
        "bias": bias,
        "iterations": iteration,
        "history": history,
        "final_loss": new_loss,
        "snapshots": snapshots,
    }


def _backtracking_line_search(
    weights: np.ndarray,
    bias: float,
    direction_w: np.ndarray,
    direction_b: float,
    grad_w: np.ndarray,
    grad_b: float,
    current_loss: float,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    initial_step: float,
    alpha: float,
    beta: float,
) -> float:
    step = initial_step
    grad_dot_dir = float(grad_w @ direction_w + grad_b * direction_b)

    while step > 1e-8:
        new_w = weights + step * direction_w
        new_b = bias + step * direction_b
        new_loss = cost_sensitive_nll(new_w, new_b, X, y, sample_weight)
        if new_loss <= current_loss + alpha * step * grad_dot_dir:
            return step
        step *= beta

    return step


def _backtracking_line_search_full(
    theta: np.ndarray,
    direction: np.ndarray,
    grad_dot_dir: float,
    current_loss: float,
    X_aug: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    initial_step: float,
    alpha: float,
    beta: float,
) -> float:
    step = initial_step
    while step > 1e-8:
        theta_new = theta + step * direction
        new_loss, _ = cost_sensitive_nll_full(theta_new, X_aug, y, sample_weight)
        if new_loss <= current_loss + alpha * step * grad_dot_dir:
            return step
        step *= beta
    return step


__all__ = [
    "GradientDescentConfig",
    "gradient_descent_cost_sensitive",
    "bfgs_cost_sensitive",
]

