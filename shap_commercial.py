"""Generate SHAP explanations for commercial solver runs using saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Module import ExperimentTracker
from Module.solver_module import configure_sklearn_like_model, solver_predict_proba
from main import explain_with_shap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explanations for commercial solver runs")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV dataset path (used to locate logs/results)")
    parser.add_argument("--method-label", type=str, required=True, choices=["commercial_gurobi", "commercial_pymoo_ga"], help="Method label used during the run")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run (directory under Result/log)")
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = ExperimentTracker(
        data_path=args.data_path,
        method_label=args.method_label,
        run_name=args.run_name,
        resume_state=Path(ExperimentTracker.compute_data_root(args.data_path)) / "log" / args.method_label / args.run_name / "state.json",
    )

    solver_results = tracker.load_solver_results()
    evaluation = tracker.load_evaluation()
    preprocessing = tracker.load_preprocessing()

    X_test = preprocessing["X_test_scaled"]
    selected_features = solver_results["solver_details"].get("selected_features", list(X_test.columns))
    X_subset = X_test[selected_features]

    if solver_results["backend"] == "custom":
        weights = solver_results["weights"]
        bias = solver_results["bias"]
        model = configure_sklearn_like_model(weights, bias, selected_features)
    else:
        model = solver_results.get("model")
        if model is None:
            raise RuntimeError("Sklearn model not found in solver results; cannot run SHAP.")

    explain_with_shap(model, X_subset, selected_features, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

