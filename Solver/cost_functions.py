"""Cost-sensitive objective functions for logistic models."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    out = np.empty_like(z, dtype=float)
    positive_mask = z >= 0
    negative_mask = ~positive_mask

    out[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    exp_z = np.exp(z[negative_mask])
    out[negative_mask] = exp_z / (1.0 + exp_z)
    return out


def cost_sensitive_nll(
    weights: np.ndarray,
    bias: float,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Compute cost-sensitive negative log-likelihood."""

    z = X @ weights + bias
    probabilities = sigmoid(z)
    eps = 1e-9
    probs = np.clip(probabilities, eps, 1.0 - eps)

    losses = -(
        y * np.log(probs)
        + (1.0 - y) * np.log(1.0 - probs)
    )
    weighted_losses = sample_weight * losses
    return float(np.sum(weighted_losses) / np.sum(sample_weight))


def cost_sensitive_nll_gradient(
    weights: np.ndarray,
    bias: float,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Gradient of the cost-sensitive NLL with respect to weights and bias."""

    z = X @ weights + bias
    probabilities = sigmoid(z)
    residual = (probabilities - y) * sample_weight

    denom = np.sum(sample_weight)
    grad_w = (X.T @ residual) / denom
    grad_b = float(np.sum(residual) / denom)
    return grad_w, grad_b


def cost_sensitive_nll_full(
    theta: np.ndarray,
    X_augmented: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Loss and gradient for combined parameter vector (weights + bias)."""

    z = X_augmented @ theta
    probabilities = sigmoid(z)
    eps = 1e-9
    probs = np.clip(probabilities, eps, 1.0 - eps)

    losses = -(
        y * np.log(probs)
        + (1.0 - y) * np.log(1.0 - probs)
    )
    weighted_losses = sample_weight * losses
    loss = float(np.sum(weighted_losses) / np.sum(sample_weight))

    residual = (probabilities - y) * sample_weight
    denom = np.sum(sample_weight)
    grad = (X_augmented.T @ residual) / denom
    return loss, grad


__all__ = [
    "sigmoid",
    "cost_sensitive_nll",
    "cost_sensitive_nll_gradient",
    "cost_sensitive_nll_full",
]

