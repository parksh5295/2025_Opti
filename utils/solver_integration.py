"""Solver integration and execution utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from Module import (
    ExperimentTracker,
    SolverConfig,
    compute_sample_weights,
    configure_sklearn_like_model,
    solve_cost_sensitive_logistic,
    solver_predict_proba,
)
from utils.convergence_plots import plot_solver_convergence
from utils.optimization_utils import optimise_threshold


def run_solver_stage(
    tracker: ExperimentTracker,
    data: Dict,
    selected_features: List[str],
    train_weights: np.ndarray,
    args,
    random_state: int,
) -> Tuple[
    Optional[np.ndarray],
    Optional[float],
    Optional[List[Dict]],
    float,
    float,
    str,
    Dict,
    object,
]:
    """Run solver optimization stage.
    
    Returns
    -------
    Tuple of (weights, bias, snapshots, threshold, val_score, solver_backend, solver_details, model).
    """
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = None
    snapshots: Optional[List[Dict[str, object]]] = None
    threshold: float
    val_score: float
    model: object

    if tracker.is_completed("solver"):
        tracker.log_event("solver", "Loading cached solver outputs")
        solver_results = tracker.load_solver_results()
        threshold = solver_results["threshold"]
        val_score = solver_results["val_score"]
        solver_backend = solver_results["backend"]
        solver_details = solver_results["solver_details"]
        snapshots = solver_results.get("snapshots")

        if solver_backend == "custom":
            weights = solver_results.get("weights")
            bias = solver_results.get("bias")
            if weights is None or bias is None:
                raise RuntimeError("Custom solver results missing weights or bias.")
            model = configure_sklearn_like_model(weights, bias, selected_features)
        else:
            model = solver_results.get("model")
            if model is None:
                raise RuntimeError(f"Solver results missing model for backend: {solver_backend}")
        
        # Plot solver convergence from saved results
        if args.plot_convergence and solver_details and "history" in solver_details:
            solver_history = solver_details["history"]
            if solver_history and "loss" in solver_history:
                try:
                    solver_plot_path = plot_solver_convergence(
                        solver_history,
                        tracker.result_dir / "solver" / "convergence.png",
                        title="Solver Convergence (from saved results)",
                        method=solver_details.get("method"),
                    )
                    tracker.log_event(
                        "solver",
                        "Generated convergence plot from saved results",
                        {"plot_path": str(solver_plot_path.relative_to(tracker.result_dir))},
                    )
                    print(f"[Solver] Convergence plot saved to: {solver_plot_path}")
                except Exception as e:
                    tracker.log_event(
                        "solver",
                        "Failed to generate convergence plot from saved results",
                        {"error": str(e)},
                        level=logging.WARNING,
                    )
    else:
        tracker.log_event("solver", "Training final classifier")
        if args.solver_backend == "custom":
            solver_config = SolverConfig(
                max_iter=args.solver_max_iter,
                learning_rate=args.solver_lr,
                tolerance=args.solver_tol,
                momentum=args.solver_momentum,
                verbose=args.solver_verbose,
                track_history=args.solver_track_history,
                method=args.solver_method,
                line_search=args.solver_line_search,
                line_search_alpha=args.solver_line_alpha,
                line_search_beta=args.solver_line_beta,
                adam_beta1=args.solver_adam_beta1,
                adam_beta2=args.solver_adam_beta2,
                adam_epsilon=args.solver_adam_epsilon,
                second_order_method=args.solver_second_order,
                track_snapshots=args.tsne_snapshots,
                snapshot_interval=max(1, args.tsne_interval),
            )

            solver_output = solve_cost_sensitive_logistic(
                data["X_train_res"][selected_features],
                data["y_train_res"],
                train_weights,
                solver_config,
            )

            weights = solver_output["weights"]
            bias = solver_output["bias"]
            snapshots = solver_output.get("snapshots")
            print(
                f"[Solver] Converged in {solver_output['iterations']} iterations | "
                f"Loss={solver_output['final_loss']:.6f}"
            )

            model = configure_sklearn_like_model(weights, bias, selected_features)
            val_probabilities = solver_predict_proba(
                data["X_val_scaled"][selected_features], weights, bias
            )
            solver_backend = "custom"
            solver_details = {
                "method": args.solver_method,
                "second_order": args.solver_second_order,
                "iterations": solver_output["iterations"],
                "final_loss": solver_output.get("final_loss"),
                "learning_rate": args.solver_lr,
                "line_search": args.solver_line_search,
                "snapshots_recorded": len(snapshots or []),
                "snapshot_interval": max(1, args.tsne_interval),
            }
            # Add history if tracked
            solver_history = None
            if args.solver_track_history and "history" in solver_output:
                solver_history = solver_output["history"]
                solver_details["history"] = solver_history
            
            # Plot solver convergence
            if args.plot_convergence and solver_history and "loss" in solver_history:
                try:
                    solver_plot_path = plot_solver_convergence(
                        solver_history,
                        tracker.result_dir / "solver" / "convergence.png",
                        title="Solver Convergence",
                        method=args.solver_method,
                    )
                    tracker.log_event(
                        "solver",
                        "Generated convergence plot",
                        {"plot_path": str(solver_plot_path.relative_to(tracker.result_dir))},
                    )
                    print(f"[Solver] Convergence plot saved to: {solver_plot_path}")
                except Exception as e:
                    tracker.log_event(
                        "solver",
                        "Failed to generate convergence plot",
                        {"error": str(e)},
                        level=logging.WARNING,
                    )
        else:
            model = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=None,
                random_state=random_state,
            )
            model.fit(
                data["X_train_res"][selected_features],
                data["y_train_res"],
                sample_weight=train_weights,
            )
            val_probabilities = model.predict_proba(
                data["X_val_scaled"][selected_features]
            )[:, 1]
            solver_backend = "sklearn"
            solver_details = {"solver": "lbfgs"}

        val_weights = compute_sample_weights(data["y_val"], args.cost_beta)
        threshold, val_score = optimise_threshold(
            data["y_val"],
            val_probabilities,
            beta=args.beta,
            sample_weight=val_weights,
        )

        tracker.save_solver_results(
            solver_backend,
            weights,
            bias,
            threshold,
            val_score,
            solver_details,
            model=model if solver_backend == "sklearn" else None,
            snapshots=snapshots if (solver_backend == "custom" and args.tsne_snapshots and snapshots) else None,
        )
        tracker.log_event(
            "solver",
            "Solver optimisation completed",
            {"threshold": threshold, "val_score": val_score, "backend": solver_backend},
        )

    return weights, bias, snapshots, threshold, val_score, solver_backend, solver_details, model

