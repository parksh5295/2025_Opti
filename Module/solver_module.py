"""Wrapper utilities that expose solver functionality to the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from Solver import GradientDescentConfig, gradient_descent_cost_sensitive, sigmoid


@dataclass
class SolverConfig:
    max_iter: int = 400
    learning_rate: float = 0.1
    tolerance: float = 1e-5
    momentum: float = 0.0
    verbose: bool = False
    track_history: bool = False


def solve_cost_sensitive_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray,
    config: SolverConfig,
) -> Dict[str, object]:
    """Solve for logistic regression weights using the custom solver."""

    X_array = X.to_numpy(dtype=float)
    y_array = y.to_numpy(dtype=float)

    gd_config = GradientDescentConfig(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        tolerance=config.tolerance,
        momentum=config.momentum,
        verbose=config.verbose,
        track_history=config.track_history,
    )

    result = gradient_descent_cost_sensitive(
        X=X_array,
        y=y_array,
        sample_weight=sample_weight.astype(float),
        config=gd_config,
    )

    return result


def solver_predict_proba(X: pd.DataFrame, weights: np.ndarray, bias: float) -> np.ndarray:
    """Compute class-1 probabilities using solver weights."""

    z = X.to_numpy(dtype=float) @ weights + bias
    return sigmoid(z)


def configure_sklearn_like_model(weights: np.ndarray, bias: float, feature_names: list[str]) -> object:
    """Create a lightweight scikit-learn compatible wrapper for reporting."""

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = weights.reshape(1, -1)
    model.intercept_ = np.array([bias])
    model.n_features_in_ = weights.size
    model.feature_names_in_ = np.array(feature_names)
    model.n_iter_ = np.array([1])
    return model


__all__ = ["SolverConfig", "solve_cost_sensitive_logistic", "solver_predict_proba", "configure_sklearn_like_model"]

