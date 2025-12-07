"""Evaluation and visualization integration utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from Module import (
    ExperimentTracker,
    EvaluationConfig,
    compute_sample_weights,
    evaluate_model,
    format_cost_sensitive_summary,
    generate_tsne_snapshots,
    solver_predict_proba,
)
from utils.explainability import explain_with_shap


def run_evaluation_stage(
    tracker: ExperimentTracker,
    data: Dict,
    selected_features: List[str],
    model: object,
    weights: Optional[np.ndarray],
    bias: Optional[float],
    threshold: float,
    solver_backend: str,
    redundancy_penalty: float,
    args,
) -> Dict:
    """Run evaluation stage.
    
    Returns
    -------
    Dictionary containing evaluation results.
    """
    if tracker.is_completed("evaluation"):
        tracker.log_event("evaluation", "Loading cached evaluation results")
        evaluation = tracker.load_evaluation()
    else:
        tracker.log_event("evaluation", "Evaluating on test data")
        if solver_backend == "custom":
            if weights is None or bias is None:
                solver_results = tracker.load_solver_results()
                weights = solver_results.get("weights")
                bias = solver_results.get("bias")
                if weights is None or bias is None:
                    raise RuntimeError("Custom solver results missing weights or bias.")
            test_probabilities = solver_predict_proba(
                data["X_test_scaled"][selected_features], weights, bias
            )
        else:
            test_probabilities = model.predict_proba(
                data["X_test_scaled"][selected_features]
            )[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)

        evaluation_config = EvaluationConfig(
            beta=args.beta,
            rho_auc=args.rho_auc,
            rho_f1=args.rho_f1,
            rho_pr=args.rho_pr,
            rho_gmean=args.rho_gmean,
            lambda_penalty=args.lambda_penalty,
            alpha_size=args.alpha_size,
        )

        evaluation = evaluate_model(
            data["y_test"],
            test_probabilities,
            test_predictions,
            evaluation_config,
            redundancy_penalty,
            len(selected_features),
        )
        tracker.save_evaluation(evaluation)
        tracker.log_event(
            "evaluation",
            "Evaluation completed",
            {
                "roc_auc": evaluation["roc_auc"],
                "pr_auc": evaluation["pr_auc"],
                "overall_score": evaluation["overall_score"],
            },
        )

    print(f"[Evaluation] Test ROC-AUC: {evaluation['roc_auc']:.4f}")
    print(f"[Evaluation] Test PR-AUC: {evaluation['pr_auc']:.4f}")
    print(
        f"[Evaluation] Cost-sensitive summary: {format_cost_sensitive_summary(evaluation['cost_sensitive'])}"
    )
    print(f"[Evaluation] Overall score: {evaluation['overall_score']:.4f}")
    print("[Evaluation] Classification report:\n" + evaluation["classification_report"])
    print("[Evaluation] Confusion matrix:\n", evaluation["confusion_matrix"])

    return evaluation


def run_visualization_stage(
    tracker: ExperimentTracker,
    data: Dict,
    selected_features: List[str],
    snapshots: Optional[List[Dict]],
    threshold: float,
    solver_backend: str,
    args,
) -> None:
    """Run visualization stage (t-SNE snapshots)."""
    if args.tsne_snapshots and solver_backend == "custom":
        if not snapshots:
            tracker.log_event(
                "visualisation",
                "t-SNE snapshots requested but solver snapshots are unavailable",
                level=logging.WARNING,
            )
        else:
            tsne_outputs = generate_tsne_snapshots(
                data["X_train_res"][selected_features],
                data["y_train_res"],
                snapshots,
                tracker.result_dir,
                threshold=threshold,
                use_adaptive_threshold=True,  # Use snapshot-specific threshold
                gif=args.tsne_gif,
                gif_duration=args.tsne_gif_duration,
            )
            if tsne_outputs:
                relative_files: List[str] = []
                for path in tsne_outputs:
                    try:
                        relative_files.append(str(path.relative_to(tracker.result_dir)))
                    except ValueError:
                        relative_files.append(str(path))
                tracker.log_event(
                    "visualisation",
                    "Generated t-SNE solver snapshots",
                    {
                        "files": relative_files,
                        "count": len(tsne_outputs),
                    },
                )
            else:
                tracker.log_event(
                    "visualisation",
                    "t-SNE snapshot generation produced no files",
                    level=logging.WARNING,
                )


def run_explainability_stage(
    tracker: ExperimentTracker,
    model: object,
    data: Dict,
    selected_features: List[str],
    skip_explainability: bool,
) -> None:
    """Run explainability stage (SHAP)."""
    if not skip_explainability:
        try:
            explain_with_shap(model, data["X_test_scaled"][selected_features], selected_features)
            tracker.log_event("explainability", "Generated SHAP explanation")
        except Exception as exc:
            tracker.log_event(
                "explainability",
                "Failed to generate SHAP explanation",
                {"error": str(exc)},
                level=logging.WARNING,
            )

