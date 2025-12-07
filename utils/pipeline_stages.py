"""Pipeline stage execution functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Module import (
    ExperimentTracker,
    GAConfig,
    GeneticFeatureSelector,
    PreprocessingConfig,
    SolverConfig,
    construct_ensemble_features,
    compute_sample_weights,
    configure_sklearn_like_model,
    evaluate_model,
    format_cost_sensitive_summary,
    generate_tsne_snapshots,
    load_and_preprocess,
    solve_cost_sensitive_logistic,
    solver_predict_proba,
)
from Module.genetic_algorithm_enhanced import (
    EnhancedGAConfig,
    EnhancedGeneticFeatureSelector,
)
from Module.genetic_algorithm_advanced import (
    AdvancedGAConfig,
    AdvancedGeneticFeatureSelector,
)
from utils.convergence_plots import (
    plot_enhanced_ga_convergence,
    plot_ga_convergence,
    plot_solver_convergence,
)
from utils.explainability import explain_with_shap
from utils.feature_selection_utils import (
    build_redundancy_penalty_matrix,
    compute_mutual_information_scores,
    compute_pca_scores,
    compute_random_forest_scores,
    compute_subset_penalty,
    information_theoretic_ensemble_scores,
    redundancy_aware_selection,
)
from utils.optimization_utils import (
    make_cost_sensitive_fitness,
    optimise_threshold,
)
from utils.transfer_learning import load_previous_best_solutions


def run_preprocessing_stage(
    tracker: ExperimentTracker,
    data_path: Path,
    preprocessing_config: PreprocessingConfig,
) -> Dict[str, pd.DataFrame]:
    """Run preprocessing stage.
    
    Returns
    -------
    Dictionary containing preprocessed dataframes.
    """
    if tracker.is_completed("preprocessing"):
        tracker.log_event("preprocessing", "Loading cached preprocessing outputs")
        data = tracker.load_preprocessing()
    else:
        tracker.log_event("preprocessing", "Starting preprocessing")
        data = load_and_preprocess(data_path, preprocessing_config)
        tracker.save_preprocessing(data)
        tracker.log_event(
            "preprocessing",
            "Completed preprocessing",
            {
                "train_resampled_rows": len(data["X_train_res"]),
                "val_rows": len(data["X_val_scaled"]),
                "test_rows": len(data["X_test_scaled"]),
            },
        )
    return data


def run_feature_scoring_stage(
    tracker: ExperimentTracker,
    data: Dict[str, pd.DataFrame],
    ensemble_weights: Dict[str, float],
    feature_ensemble_mode: str,
    feature_ensemble_top_k: int,
    random_state: int,
) -> Tuple[Dict[str, float], Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, object]]]:
    """Run feature scoring stage.
    
    Returns
    -------
    Tuple of (ensemble_scores, constructed_frames, constructed_metadata).
    """
    if tracker.is_completed("feature_scoring"):
        tracker.log_event("feature_scoring", "Loading cached feature scores")
        scores_payload = tracker.load_feature_scores()
        score_map = scores_payload["scores"]
        ensemble_scores = score_map.get("ensemble", {})
        
        constructed_frames = scores_payload.get("constructed_frames") or {}
        metadata_fs = scores_payload.get("metadata", {})
        stored_mode = metadata_fs.get("ensemble_mode")
        if stored_mode and stored_mode != feature_ensemble_mode:
            tracker.log_event(
                "feature_scoring",
                "Stored ensemble mode differs from requested mode; continuing with stored configuration.",
                {"stored_mode": stored_mode, "requested_mode": feature_ensemble_mode},
                level=logging.WARNING,
            )
        if constructed_frames:
            for key, frame in constructed_frames.items():
                data[key] = pd.concat([data[key], frame], axis=1)
        constructed_metadata = {
            "created_columns": metadata_fs.get("constructed_columns", []),
            "top_features": metadata_fs.get("constructed_top_features", {}),
            "top_k": metadata_fs.get("constructed_top_k"),
            "new_scores": metadata_fs.get("constructed_new_scores", {}),
        }
        new_scores = constructed_metadata.get("new_scores") or {}
        if new_scores:
            ensemble_scores.update({k: float(v) for k, v in new_scores.items()})
        return ensemble_scores, constructed_frames, constructed_metadata
    else:
        tracker.log_event("feature_scoring", "Computing feature importance scores")
        pca_scores = compute_pca_scores(data["X_train_res"], random_state=random_state)
        mi_scores = compute_mutual_information_scores(
            data["X_train_res"],
            data["y_train_res"],
            random_state=random_state,
        )
        rf_scores = compute_random_forest_scores(
            data["X_train_res"],
            data["y_train_res"],
            random_state=random_state,
        )

        ensemble_scores = information_theoretic_ensemble_scores(
            {"pca": pca_scores, "mutual_info": mi_scores, "random_forest": rf_scores},
            ensemble_weights,
        )
        constructed_frames = None
        constructed_metadata = None
        if feature_ensemble_mode == "construct":
            constructed_frames, constructed_metadata = construct_ensemble_features(
                {
                    key: data[key]
                    for key in [
                        "X_train_res",
                        "X_train_scaled",
                        "X_val_scaled",
                        "X_test_scaled",
                    ]
                },
                pca_scores,
                mi_scores,
                rf_scores,
                ensemble_scores,
                top_k=feature_ensemble_top_k,
            )
            if constructed_frames:
                for key, frame in constructed_frames.items():
                    data[key] = pd.concat([data[key], frame], axis=1)
            if constructed_metadata:
                new_scores = constructed_metadata.get("new_scores", {})
                if new_scores:
                    ensemble_scores.update({k: float(v) for k, v in new_scores.items()})
                tracker.log_event(
                    "feature_scoring",
                    "Constructed ensemble features",
                    {
                        "created_columns": len(constructed_metadata.get("created_columns", [])),
                        "top_k": constructed_metadata.get("top_k"),
                    },
                )
        tracker.save_feature_scores(
            pca_scores,
            mi_scores,
            rf_scores,
            ensemble_scores,
            ensemble_mode=feature_ensemble_mode,
            constructed_frames=constructed_frames or None,
            constructed_metadata=constructed_metadata,
        )
        tracker.log_event(
            "feature_scoring",
            "Feature scoring completed",
            {"num_features": len(ensemble_scores)},
        )
        return ensemble_scores, constructed_frames, constructed_metadata


def run_redundancy_stage(
    tracker: ExperimentTracker,
    data: Dict[str, pd.DataFrame],
    ensemble_scores: Dict[str, float],
    penalty_weights: Dict[str, float],
    redundancy_budget: float,
    min_candidate_features: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run redundancy minimization stage.
    
    Returns
    -------
    Tuple of (penalty_matrix, candidate_features).
    """
    if tracker.is_completed("redundancy"):
        tracker.log_event("redundancy", "Loading cached redundancy analysis")
        redundancy_data = tracker.load_redundancy()
        penalty_matrix = redundancy_data["penalty_matrix"]
        candidate_features = redundancy_data["candidate_features"]
    else:
        tracker.log_event("redundancy", "Computing redundancy penalty matrix")
        penalty_matrix = build_redundancy_penalty_matrix(
            data["X_train_res"],
            data["y_train_res"],
            penalty_weights,
        )

        candidate_features = redundancy_aware_selection(
            ensemble_scores,
            penalty_matrix,
            ensemble_scores,
            budget=redundancy_budget,
            min_features=min_candidate_features,
        )
        tracker.save_redundancy(penalty_matrix, candidate_features)
        tracker.log_event(
            "redundancy",
            "Candidate feature set generated",
            {"num_candidates": len(candidate_features)},
        )
    return penalty_matrix, candidate_features

