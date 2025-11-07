"""Generate SHAP explanations for commercial solver runs using saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from Module.solver_module import configure_sklearn_like_model
from main import explain_with_shap
from Module.experiment_tracker import ExperimentTracker


def load_state(state_path: Path) -> Dict[str, Any]:
    with state_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_artifact(result_dir: Path, meta: Dict[str, Any]):
    path = result_dir / meta["path"]
    typ = meta["type"]
    if typ == "pandas_pickle":
        return pd.read_pickle(path)
    if typ == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if typ == "joblib":
        from joblib import load

        return load(path)
    if typ == "npy":
        return np.load(path)
    raise ValueError(f"Unsupported artifact type: {typ}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explanations for commercial solver runs")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV dataset path (used to locate logs/results)")
    parser.add_argument("--method-label", type=str, required=True, choices=["commercial_gurobi", "commercial_pymoo_ga"], help="Method label used during the run")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run (directory under Result/log)")
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = ExperimentTracker.compute_data_root(Path(args.data_path))

    state_path = base_root / "log" / args.method_label / args.run_name / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    state = load_state(state_path)
    stages = state.get("stages", {})
    if "solver" not in stages or "evaluation" not in stages:
        raise RuntimeError(
            "Solver or evaluation stage not found. Re-run `main_commercial.py` for the same run to produce the necessary artifacts."
        )

    result_dir = Path(state["result_dir"])

    preprocessing_artifacts = stages["preprocessing"]["artifacts"]
    X_test_meta = preprocessing_artifacts.get("X_test_scaled")
    if X_test_meta is None:
        raise RuntimeError("X_test_scaled artifact not found in preprocessing stage.")
    X_test = load_artifact(result_dir, X_test_meta)

    solver_stage = stages["solver"]
    solver_artifacts = solver_stage["artifacts"]
    solver_metadata = solver_stage["metadata"]
    solver_details = json.loads((result_dir / solver_artifacts["solver_details"]["path"]).read_text(encoding="utf-8"))

    selected_features = solver_details.get("selected_features", list(X_test.columns))
    X_subset = X_test[selected_features]

    weights_meta = solver_artifacts.get("weights")
    bias_meta_entry = solver_artifacts.get("bias")
    model_meta = solver_artifacts.get("model")

    if weights_meta and bias_meta_entry:
        weights = load_artifact(result_dir, weights_meta)
        bias_meta = load_artifact(result_dir, bias_meta_entry)
        bias = float(bias_meta["bias"])
        model = configure_sklearn_like_model(weights, bias, selected_features)
    elif model_meta is not None:
        model = load_artifact(result_dir, model_meta)
    else:
        raise RuntimeError("No compatible weights or model artifacts found for SHAP explanation.")

    explain_with_shap(model, X_subset, selected_features, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

