"""Wrapper for running pymoo-based evolutionary optimisation as a commercial-style interface."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


class LogisticSubsetProblem(ElementwiseProblem):
    def __init__(self, X: np.ndarray, y: np.ndarray, cost_beta: float, **kwargs) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.cost_beta = float(cost_beta)
        super().__init__(n_var=self.X.shape[1], n_obj=1, xl=0, xu=1, type_var=np.bool_, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if not np.any(x):
            out["F"] = 1e6
            return
        selected = self.X[:, x]
        weights = np.where(self.y == 1, self.cost_beta, 1.0)
        logits = selected.sum(axis=1)
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-9
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.mean(weights * (self.y * np.log(probs) + (1 - self.y) * np.log(1 - probs)))
        out["F"] = loss


def run_pymoo_ga(
    X: np.ndarray,
    y: np.ndarray,
    cost_beta: float,
    population_size: int = 50,
    generations: int = 50,
) -> Dict[str, object]:
    problem = LogisticSubsetProblem(X, y, cost_beta)
    algorithm = NSGA2(pop_size=population_size)
    result = minimize(problem, algorithm, termination=("n_gen", generations), verbose=False)
    best_mask = result.X.astype(bool)
    return {
        "mask": best_mask,
        "loss": float(result.F[0]) if result.F is not None else None,
    }


__all__ = ["run_pymoo_ga"]

