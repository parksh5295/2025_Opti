"""Entry point for running experiments with commercial solvers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Module import (
    ExperimentTracker,
    PreprocessingConfig,
    compute_sample_weights,
    load_and_preprocess,
)
from Commercial_Solver.Numerical_optimization.gurobi_solver import solve_with_gurobi
from Commercial_Solver.Metaheuristic_based.pymoo_ga import run_pymoo_ga


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commercial solver optimisation pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--target-column", type=str, default="Class")
    parser.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "pymoo_ga"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--force-new-run", action="store_true")
    parser.add_argument("--cost-beta", type=float, default=5.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    method_label = f"commercial_{args.solver}"
    resume_state = args.resume_from
    if resume_state is None and not args.force_new_run:
        auto = ExperimentTracker.find_latest_state(args.data_path, method_label)
        if auto is not None:
            print(f"[Tracker] Resuming from {auto}")
            resume_state = auto

    tracker = ExperimentTracker(
        data_path=args.data_path,
        method_label=method_label,
        run_name=args.run_name,
        resume_state=resume_state,
    )

    preprocessing_config = PreprocessingConfig(
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        use_smote=True,
        use_undersampling=False,
        beta_cost=args.cost_beta,
    )

    try:
        if tracker.is_completed("preprocessing"):
            data = tracker.load_preprocessing()
        else:
            data = load_and_preprocess(args.data_path, preprocessing_config)
            tracker.save_preprocessing(data)

        X_train = data["X_train_res"].to_numpy()
        y_train = data["y_train_res"].to_numpy()

        if args.solver == "gurobi":
            result = solve_with_gurobi(
                X_train,
                y_train,
                compute_sample_weights(data["y_train_res"], args.cost_beta),
            )
        else:
            result = run_pymoo_ga(
                X_train,
                y_train,
                cost_beta=args.cost_beta,
            )

        tracker.log_event(
            "commercial_solver",
            "Solver run completed",
            {"solver": args.solver, **{k: v for k, v in result.items() if k != "mask"}},
        )
        tracker.mark_status("completed")
        print(f"[Tracker] Logs saved to {tracker.log_dir}")
    except Exception as exc:
        tracker.log_event("commercial_solver", "Solver run failed", {"error": str(exc)}, level=40)
        tracker.mark_status("failed")
        raise


if __name__ == "__main__":
    run()

