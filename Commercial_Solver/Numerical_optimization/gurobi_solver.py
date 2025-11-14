"""Thin wrapper for solving logistic regression with a cost-sensitive objective using Gurobi.

Note: This module assumes that `gurobipy` is installed and a valid Gurobi
license is configured on the host machine.  The implementation mirrors the
custom solver but relies on Gurobi's quadratic programming capabilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from gurobipy import GRB, Model


def solve_with_gurobi(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    max_iter: int = 200,
    name: str = "gurobi_logistic",
    track_snapshots: bool = False,
    snapshot_interval: int = 5,
) -> Dict[str, object]:
    n_samples, n_features = X.shape
    model = Model(name)
    model.Params.OutputFlag = 0
    model.Params.IterationLimit = max_iter

    w = model.addVars(n_features, lb=-GRB.INFINITY, name="w")
    b = model.addVar(lb=-GRB.INFINITY, name="b")

    loss_terms = []
    for i in range(n_samples):
        margin = sum(w[j] * float(X[i, j]) for j in range(n_features)) + b
        slack_pos = model.addVar(lb=0.0, name=f"slack_pos_{i}")
        slack_neg = model.addVar(lb=0.0, name=f"slack_neg_{i}")
        model.addConstr(margin >= -slack_neg)
        model.addConstr(margin <= slack_pos)
        weight = float(sample_weight[i])
        if int(y[i]) == 1:
            loss_terms.append(weight * slack_neg)
        else:
            loss_terms.append(weight * slack_pos)

    model.setObjective(sum(loss_terms) / np.sum(sample_weight), GRB.MINIMIZE)
    
    snapshots: List[Dict[str, object]] = []
    last_snapshot_iter = -1
    
    def callback(model_obj, where):
        """Callback to track iterations."""
        nonlocal last_snapshot_iter
        
        if track_snapshots and where == GRB.Callback.BARRIER:
            # Get current iteration count
            try:
                itcnt = model_obj.cbGet(GRB.Callback.BARRIER_ITRCNT)
                
                # Save snapshot at specified intervals
                if itcnt == 0 or itcnt % snapshot_interval == 0 or itcnt == max_iter:
                    if itcnt != last_snapshot_iter:
                        # Try to get current solution
                        try:
                            current_w = np.array([model_obj.cbGetSolution(w[j]) for j in range(n_features)])
                            current_b = model_obj.cbGetSolution(b)
                            
                            snapshots.append({
                                "iteration": int(itcnt),
                                "weights": current_w.tolist(),
                                "bias": float(current_b),
                            })
                            last_snapshot_iter = itcnt
                        except Exception:
                            # Solution might not be available at this point
                            pass
            except Exception:
                # BARRIER callback might not be available for all solvers
                pass
    
    model.optimize(callback if track_snapshots else None)

    weights = np.array([float(w[j].X) for j in range(n_features)])
    bias = float(b.X)

    return_dict = {
        "weights": weights,
        "bias": bias,
        "status": model.Status,
        "objective": model.objVal if model.SolCount else None,
    }
    
    if track_snapshots and snapshots:
        # Add final snapshot if not already included
        if not snapshots or snapshots[-1]["iteration"] != max_iter:
            snapshots.append({
                "iteration": max_iter,
                "weights": weights.tolist(),
                "bias": bias,
            })
        return_dict["snapshots"] = snapshots
    
    return return_dict


__all__ = ["solve_with_gurobi"]

