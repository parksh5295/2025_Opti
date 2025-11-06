"""Wrapper utilities that expose solver functionality to the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from Solver import (
    GradientDescentConfig,
    bfgs_cost_sensitive,
    gradient_descent_cost_sensitive,
    sigmoid,
)


@dataclass
class SolverConfig:
    max_iter: int = 400
    learning_rate: float = 0.1
    tolerance: float = 1e-5
    momentum: float = 0.0
    verbose: bool = False
    track_history: bool = False
    method: str = "gd"
    line_search: bool = False
    line_search_alpha: float = 0.3
    line_search_beta: float = 0.8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    second_order_method: str = "none"  # "none" or "bfgs"


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
        method=config.method,
        line_search=config.line_search,
        line_search_alpha=config.line_search_alpha,
        line_search_beta=config.line_search_beta,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        use_second_order=config.second_order_method.lower() != "none",
        second_order_method=config.second_order_method.lower(),
    )

    if gd_config.use_second_order and gd_config.second_order_method == "bfgs":
        result = bfgs_cost_sensitive(
            X=X_array,
            y=y_array,
            sample_weight=sample_weight.astype(float),
            config=gd_config,
        )
    else:
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

