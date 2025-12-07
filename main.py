"""Information-theoretic ensemble pipeline for credit-card fraud detection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Module import ExperimentTracker, PreprocessingConfig, compute_sample_weights
from utils.argument_parsers import create_main_parser
from utils.evaluation_integration import (
    run_evaluation_stage,
    run_explainability_stage,
    run_visualization_stage,
)
from utils.feature_selection_utils import compute_subset_penalty
from utils.ga_integration import load_ga_from_reuse, run_ga_stage
from utils.pipeline_stages import (
    run_feature_scoring_stage,
    run_preprocessing_stage,
    run_redundancy_stage,
)
from utils.solver_integration import run_solver_stage


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete fraud detection pipeline."""
    random_state = args.random_state
    use_second_order = (
        args.solver_backend == "custom" and args.solver_second_order == "bfgs"
    )
    method_label = "with_hessian" if use_second_order else "without_hessian"

    # Initialize tracker
    resume_state = args.resume_from
    run_name = args.run_name
    if resume_state is None and not args.force_new_run:
        auto_state = ExperimentTracker.find_latest_state(args.data_path, method_label)
        if auto_state is not None:
            print(f"[Tracker] Resuming from state file: {auto_state}")
            resume_state = auto_state
            run_name = None
        elif args.run_name is None:
            print("[Tracker] No existing run to resume; starting a new run.")

    tracker = ExperimentTracker(
        data_path=args.data_path,
        method_label=method_label,
        run_name=run_name,
        resume_state=resume_state,
    )

    tracker.log_event(
        "pipeline",
        "Run initialised",
        {
            "solver_backend": args.solver_backend,
            "solver_method": args.solver_method,
            "second_order": args.solver_second_order,
        },
    )

    preprocessing_config = PreprocessingConfig(
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=random_state,
        use_smote=not args.no_smote,
        use_undersampling=args.enable_undersampling,
        undersample_ratio=args.undersample_ratio,
        beta_cost=args.cost_beta,
    )

    # Handle GA reuse if requested (before running pipeline stages)
    reuse_ga_data = None
    if args.reuse_ga_run:
        try:
            reuse_ga_data = load_ga_from_reuse(
                tracker,
                args.reuse_ga_run,
                method_label,
                args.data_path,
                None,  # feature_columns not available yet, will be set later
            )
        except Exception as exc:
            tracker.log_event(
                "ga",
                "Failed to load GA data for reuse",
                {"error": str(exc)},
                level=logging.WARNING,
            )
            raise

    try:
        # ------------------------------------------------------------------
        # Stage: Preprocessing
        # ------------------------------------------------------------------
        data = run_preprocessing_stage(tracker, args.data_path, preprocessing_config)
        feature_columns = list(data["X_train_res"].columns)

        # ------------------------------------------------------------------
        # Stage: Feature Scoring
        # ------------------------------------------------------------------
        ensemble_weights = {
            "pca": args.weight_pca,
            "mutual_info": args.weight_mi,
            "random_forest": args.weight_rf,
        }
        ensemble_scores, constructed_frames, constructed_metadata = run_feature_scoring_stage(
            tracker,
            data,
            ensemble_weights,
            args.feature_ensemble_mode,
            args.feature_ensemble_top_k,
            random_state,
        )

        # ------------------------------------------------------------------
        # Stage: Redundancy minimisation
        # ------------------------------------------------------------------
        penalty_weights = {
            "cmi": args.penalty_weight_cmi,
            "corr": args.penalty_weight_corr,
            "vif": args.penalty_weight_vif,
        }
        penalty_matrix, candidate_features = run_redundancy_stage(
            tracker,
            data,
            ensemble_scores,
            penalty_weights,
            args.redundancy_budget,
            args.min_candidate_features,
        )

        print(
            f"[Pipeline] Candidate features after redundancy-aware selection: {len(candidate_features)}"
        )

        # ------------------------------------------------------------------
        # Stage: Genetic Algorithm
        # ------------------------------------------------------------------
        # Update reuse_ga_data with feature_columns if available
        if reuse_ga_data is not None:
            feature_columns = list(data["X_train_res"].columns)
        
        selected_features, ga_score, redundancy_penalty, ga_history, diversity_history = run_ga_stage(
            tracker,
            data,
            candidate_features,
            ensemble_scores,
            penalty_matrix,
            ensemble_weights,
            preprocessing_config,
            args,
            random_state,
            reuse_ga_data=reuse_ga_data,
        )

        if redundancy_penalty is None:
            redundancy_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_weights)

        print(f"[GA] Selected features ({len(selected_features)}): {selected_features}")
        if ga_score is not None:
            print(f"[GA] Best fitness score: {ga_score:.4f}")
        print(f"[GA] Redundancy penalty: {redundancy_penalty:.4f}")

        # ------------------------------------------------------------------
        # Stage: Solver optimisation
        # ------------------------------------------------------------------
        train_weights = compute_sample_weights(data["y_train_res"], args.cost_beta)
        weights, bias, snapshots, threshold, val_score, solver_backend, solver_details, model = run_solver_stage(
            tracker,
            data,
            selected_features,
            train_weights,
            args,
            random_state,
        )

        print(f"[Solver] Optimal decision threshold: {threshold:.3f} (F{args.beta:.1f}={val_score:.4f})")

        # ------------------------------------------------------------------
        # Stage: Evaluation
        # ------------------------------------------------------------------
        evaluation = run_evaluation_stage(
            tracker,
            data,
            selected_features,
            model,
            weights,
            bias,
            threshold,
            solver_backend,
            redundancy_penalty,
            args,
        )

        # ------------------------------------------------------------------
        # Stage: Visualization
        # ------------------------------------------------------------------
        run_visualization_stage(
            tracker,
            data,
            selected_features,
            snapshots,
            threshold,
            solver_backend,
            args,
        )

        # ------------------------------------------------------------------
        # Stage: Explainability
        # ------------------------------------------------------------------
        run_explainability_stage(
            tracker,
            model,
            data,
            selected_features,
            args.skip_explainability,
        )

        tracker.mark_status("completed")
        tracker.log_event("pipeline", "Run completed successfully")
        print(f"[Tracker] Results saved to {tracker.result_dir}")
        print(f"[Tracker] Logs saved to {tracker.log_dir}")

    except KeyboardInterrupt:
        tracker.log_event("pipeline", "Run interrupted by user", level=logging.WARNING)
        tracker.mark_status("interrupted")
        raise
    except Exception as exc:
        tracker.log_event(
            "pipeline",
            "Run failed",
            {"error": str(exc)},
            level=logging.ERROR,
        )
        tracker.mark_status("failed")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for main.py."""
    parser = create_main_parser()
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()

