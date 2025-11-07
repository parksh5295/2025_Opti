"""Entry point for running experiments with commercial solvers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

from Module import (
    EvaluationConfig,
    ExperimentTracker,
    PreprocessingConfig,
    compute_sample_weights,
    evaluate_model,
    format_cost_sensitive_summary,
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
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--rho-auc", type=float, default=0.35)
    parser.add_argument("--rho-f1", type=float, default=0.25)
    parser.add_argument("--rho-pr", type=float, default=0.2)
    parser.add_argument("--rho-gmean", type=float, default=0.1)
    return parser.parse_args()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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

        X_train_df = data["X_train_res"]
        X_val_df = data["X_val_scaled"]
        X_test_df = data["X_test_scaled"]
        y_train = data["y_train_res"].to_numpy()
        y_val = data["y_val"].to_numpy()
        y_test = data["y_test"].to_numpy()

        feature_columns = list(X_train_df.columns)

        if args.solver == "gurobi":
            result = solve_with_gurobi(
                X_train_df.to_numpy(),
                y_train,
                compute_sample_weights(data["y_train_res"], args.cost_beta),
            )
            weights = result["weights"]
            bias = result["bias"]
            selected_columns = [col for col, w in zip(feature_columns, weights) if abs(w) > 1e-9]
            if not selected_columns:
                selected_columns = feature_columns
            model_probs_val = _sigmoid(X_val_df[selected_columns].to_numpy() @ weights[: len(selected_columns)] + bias)
            model_probs_test = _sigmoid(X_test_df[selected_columns].to_numpy() @ weights[: len(selected_columns)] + bias)
            backend = "gurobi"
            solver_details = {"status": result.get("status"), "objective": result.get("objective")}
            sklearn_model = None
        else:
            ga_result = run_pymoo_ga(
                X_train_df.to_numpy(),
                y_train,
                cost_beta=args.cost_beta,
            )
            mask = np.asarray(ga_result.get("mask", []), dtype=bool)
            if mask.size != len(feature_columns):
                mask = np.ones(len(feature_columns), dtype=bool)
            selected_columns = [col for col, flag in zip(feature_columns, mask) if flag]
            if not selected_columns:
                selected_columns = feature_columns
            logistic = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.random_state)
            logistic.fit(
                X_train_df[selected_columns],
        y_train,
                sample_weight=compute_sample_weights(data["y_train_res"], args.cost_beta),
            )
            model_probs_val = logistic.predict_proba(X_val_df[selected_columns])[:, 1]
            model_probs_test = logistic.predict_proba(X_test_df[selected_columns])[:, 1]
            weights = logistic.coef_.ravel()
            bias = float(logistic.intercept_[0])
            backend = "pymoo_ga"
            solver_details = {"loss": ga_result.get("loss")}
            sklearn_model = logistic
            tracker.log_event(
                "commercial_solver",
                "Selected features from GA",
                {"count": len(selected_columns), "features": selected_columns},
            )

        val_threshold = 0.5
        val_preds = (model_probs_val >= val_threshold).astype(int)
        val_score = fbeta_score(
            y_val,
            val_preds,
            beta=args.beta,
            sample_weight=compute_sample_weights(data["y_val"], args.cost_beta),
            zero_division=0,
        )

        evaluation_config = EvaluationConfig(
            beta=args.beta,
            rho_auc=args.rho_auc,
            rho_f1=args.rho_f1,
            rho_pr=args.rho_pr,
            rho_gmean=args.rho_gmean,
        )
        evaluation = evaluate_model(
            data["y_test"],
            model_probs_test,
            test_predictions,
            evaluation_config,
            redundancy_penalty=0.0,
            subset_size=len(selected_columns),
        )

        tracker.save_solver_results(
            backend,
            weights if backend == "gurobi" else None,
            bias if backend == "gurobi" else None,
            threshold=val_threshold,
            val_score=val_score,
            solver_details={"selected_features": selected_columns, **solver_details},
            model=sklearn_model,
        )
        tracker.save_evaluation(evaluation)
        tracker.log_event(
            "evaluation",
            "Evaluation completed",
            {
                "roc_auc": evaluation["roc_auc"],
                "pr_auc": evaluation["pr_auc"],
                "overall_score": evaluation["overall_score"],
                "selected_features": selected_columns,
            },
        )
        tracker.mark_status("completed")

        print(f"[Evaluation] Test ROC-AUC: {evaluation['roc_auc']:.4f}")
        print(f"[Evaluation] Test PR-AUC: {evaluation['pr_auc']:.4f}")
        print(
            f"[Evaluation] Cost-sensitive summary: {format_cost_sensitive_summary(evaluation['cost_sensitive'])}"
        )
        print(f"[Evaluation] Overall score: {evaluation['overall_score']:.4f}")
        print("[Evaluation] Classification report:\n" + evaluation["classification_report"])
        print("[Evaluation] Confusion matrix:\n", evaluation["confusion_matrix"])
        print(f"[Tracker] Results saved to {tracker.result_dir}")
        print(f"[Tracker] Logs saved to {tracker.log_dir}")
    except Exception as exc:
        tracker.log_event("commercial_solver", "Solver run failed", {"error": str(exc)}, level=40)
        tracker.mark_status("failed")
        raise


if __name__ == "__main__":
    run()

