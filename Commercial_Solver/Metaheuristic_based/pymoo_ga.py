"""Wrapper for running pymoo-based evolutionary optimisation as a commercial-style interface."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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
        mask = np.asarray(x).astype(bool)
        if not np.any(mask):
            out["F"] = 1e6
            return
        selected = self.X[:, mask]
        weights = np.where(self.y == 1, self.cost_beta, 1.0)
        logits = selected.sum(axis=1)
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-9
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.mean(weights * (self.y * np.log(probs) + (1 - self.y) * np.log(1 - probs)))
        out["F"] = loss


class GenerationCallback:
    """Callback to track generations and save snapshots for t-SNE visualization."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, cost_beta: float, snapshot_interval: int = 5):
        self.X = X
        self.y = y
        self.cost_beta = cost_beta
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[Dict[str, object]] = []
        self._last_gen = -1
    
    def __call__(self, algorithm):
        """Called at each generation (pymoo callback interface)."""
        gen = algorithm.n_gen
        
        # Save snapshot at specified intervals
        max_gen = getattr(algorithm.termination, 'n_max_gen', None)
        if max_gen is None:
            # Try to get from termination tuple
            if hasattr(algorithm.termination, 'n_max_gen'):
                max_gen = algorithm.termination.n_max_gen
            else:
                max_gen = gen + 1  # Fallback
        
        if gen == 0 or gen % self.snapshot_interval == 0 or (max_gen and gen >= max_gen):
            # Get best individual
            pop = algorithm.pop
            F = pop.get("F")
            X_pop = pop.get("X")
            
            if F is not None and len(F) > 0:
                # Find best (minimum loss)
                best_idx = np.argmin(F)
                best_mask = X_pop[best_idx].astype(bool)
                
                # Train logistic regression on selected features
                if np.any(best_mask):
                    from sklearn.linear_model import LogisticRegression
                    selected_X = self.X[:, best_mask]
                    sample_weights = np.where(self.y == 1, self.cost_beta, 1.0)
                    
                    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
                    model.fit(selected_X, self.y, sample_weight=sample_weights)
                    
                    weights = np.zeros(self.X.shape[1])
                    weights[best_mask] = model.coef_.ravel()
                    bias = float(model.intercept_[0])
                    
                    self.snapshots.append({
                        "generation": gen,
                        "weights": weights.tolist(),
                        "bias": bias,
                        "selected_features": best_mask.tolist(),
                        "loss": float(F[best_idx]),
                    })
        
        self._last_gen = gen


def run_pymoo_ga(
    X: np.ndarray,
    y: np.ndarray,
    cost_beta: float,
    population_size: int = 50,
    generations: int = 50,
    track_snapshots: bool = False,
    snapshot_interval: int = 5,
) -> Dict[str, object]:
    problem = LogisticSubsetProblem(X, y, cost_beta)
    algorithm = NSGA2(pop_size=population_size)
    
    callback = None
    if track_snapshots:
        callback = GenerationCallback(X, y, cost_beta, snapshot_interval)
    
    # Only pass callback if it's not None (pymoo doesn't handle None callbacks well)
    minimize_kwargs = {
        "problem": problem,
        "algorithm": algorithm,
        "termination": ("n_gen", generations),
        "verbose": False,
    }
    if callback is not None:
        minimize_kwargs["callback"] = callback
    
    result = minimize(**minimize_kwargs)
    
    best_mask = result.X.astype(bool)
    
    return_dict = {
        "mask": best_mask,
        "loss": float(result.F[0]) if result.F is not None else None,
    }
    
    if track_snapshots and callback:
        # Convert snapshots to iteration format for compatibility
        snapshots = []
        for snap in callback.snapshots:
            snapshots.append({
                "iteration": snap["generation"],
                "weights": snap["weights"],
                "bias": snap["bias"],
            })
        return_dict["snapshots"] = snapshots
    
    return return_dict


__all__ = ["run_pymoo_ga"]

