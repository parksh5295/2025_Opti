"""Optimization utility functions.

This module contains functions for cost-sensitive optimization, fitness functions,
and threshold optimization used in the genetic algorithm and solver stages.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

from utils.feature_selection_utils import compute_subset_penalty


def cost_sensitive_negative_log_likelihood(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sample_weight: ArrayLike,
) -> float:
    """Compute cost-sensitive negative log-likelihood."""
    eps = 1e-9
    y_prob = np.clip(y_prob, eps, 1 - eps)
    y_true = np.asarray(y_true)
    weights = np.asarray(sample_weight)
    nll = -np.sum(
        weights
        * (
            y_true * np.log(y_prob)
            + (1 - y_true) * np.log(1 - y_prob)
        )
    )
    return nll / np.sum(weights)


def make_cost_sensitive_fitness(
    feature_names: Sequence[str],
    ensemble_weights: Dict[str, float],
    penalty_matrix: pd.DataFrame,
    lambda_penalty: float,
    alpha_size: float,
    cost_beta: float,
    random_state: int,
    cv_splits: int = 5,
) -> Callable:
    """Create a cost-sensitive fitness function for genetic algorithm."""
    feature_names = list(feature_names)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def fitness(chromosome: np.ndarray, X: np.ndarray, y: ArrayLike) -> float:
        indices = np.where(chromosome == 1)[0]
        if indices.size == 0:
            return -np.inf

        selected_features = [feature_names[idx] for idx in indices]
        subset_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_weights)

        redundancy_term = lambda_penalty * subset_penalty
        size_term = alpha_size * len(selected_features)

        X_subset = X[:, indices]
        y_array = np.asarray(y)

        total_nll = 0.0
        for train_idx, val_idx in skf.split(X_subset, y_array):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            model = LogisticRegression(
                max_iter=1500,
                solver="lbfgs",
                class_weight=None,
                random_state=random_state,
            )

            sample_weight_train = np.where(y_train == 1, cost_beta, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight_train)

            probabilities = model.predict_proba(X_val)[:, 1]
            sample_weight_val = np.where(y_val == 1, cost_beta, 1.0)
            total_nll += cost_sensitive_negative_log_likelihood(y_val, probabilities, sample_weight_val)

        avg_nll = total_nll / cv_splits
        fitness_score = -(avg_nll + redundancy_term + size_term)
        return fitness_score

    return fitness


def optimise_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    beta: float = 2.0,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float]:
    """Optimize decision threshold using F-beta score."""
    def objective(threshold: float) -> float:
        preds = (y_prob >= threshold).astype(int)
        score = fbeta_score(
            y_true,
            preds,
            beta=beta,
            zero_division=0,
            sample_weight=sample_weight,
        )
        return -score

    result = minimize_scalar(
        objective,
        bounds=(0.01, 0.99),
        method="bounded",
        options={"xatol": 1e-3},
    )

    best_threshold = float(result.x)
    best_score = float(-result.fun)
    return best_threshold, best_score

