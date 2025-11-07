"""Thin wrapper for solving logistic regression with a cost-sensitive objective using Gurobi.

Note: This module assumes that `gurobipy` is installed and a valid Gurobi
license is configured on the host machine.  The implementation mirrors the
custom solver but relies on Gurobi's quadratic programming capabilities.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from gurobipy import GRB, Model


def solve_with_gurobi(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    max_iter: int = 200,
    name: str = "gurobi_logistic",
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
    model.optimize()

    weights = np.array([float(w[j].X) for j in range(n_features)])
    bias = float(b.X)

    return {
        "weights": weights,
        "bias": bias,
        "status": model.Status,
        "objective": model.objVal if model.SolCount else None,
    }


__all__ = ["solve_with_gurobi"]

