"""Prepare GA features cache: Preprocessing → Feature Scoring → Redundancy → GA.

This script runs the common, time-consuming stages (preprocessing through GA)
and saves the results to a fixed cache location that can be reused across
different solver runs.

The cache is keyed by:
- Data path
- Preprocessing configuration (test_size, validation_size, random_state, SMOTE, etc.)
- Feature scoring configuration (ensemble weights, ensemble mode, top_k)
- Redundancy configuration (penalty weights, budget, min_candidates)
- GA configuration (population, generations, mutation, etc.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from Module import (
    ExperimentTracker,
    GAConfig,
    GeneticFeatureSelector,
    PreprocessingConfig,
    compute_sample_weights,
    construct_ensemble_features,
    load_and_preprocess,
)
from utils.feature_selection_utils import (
    build_redundancy_penalty_matrix,
    compute_mutual_information_scores,
    compute_pca_scores,
    compute_random_forest_scores,
    compute_subset_penalty,
    information_theoretic_ensemble_scores,
    redundancy_aware_selection,
)
from utils.optimization_utils import make_cost_sensitive_fitness
from sklearn.linear_model import LogisticRegression


def compute_cache_key(
    data_path: Path,
    preprocessing_config: PreprocessingConfig,
    ensemble_weights: Dict[str, float],
    ensemble_mode: str,
    ensemble_top_k: Optional[int],
    penalty_weights: Dict[str, float],
    redundancy_budget: float,
    min_candidate_features: int,
    ga_config: GAConfig,
    lambda_penalty: float,
    alpha_size: float,
    cost_beta: float,
    ga_cv_splits: int,
) -> str:
    """Compute a hash-based cache key from configuration parameters."""
    config_dict = {
        "data_path": str(data_path.resolve()),
        "preprocessing": {
            "target_column": preprocessing_config.target_column,
            "test_size": preprocessing_config.test_size,
            "validation_size": preprocessing_config.validation_size,
            "random_state": preprocessing_config.random_state,
            "use_smote": preprocessing_config.use_smote,
            "use_undersampling": preprocessing_config.use_undersampling,
            "undersample_ratio": preprocessing_config.undersample_ratio,
            "beta_cost": preprocessing_config.beta_cost,
        },
        "feature_scoring": {
            "ensemble_weights": ensemble_weights,
            "ensemble_mode": ensemble_mode,
            "ensemble_top_k": ensemble_top_k,
        },
        "redundancy": {
            "penalty_weights": penalty_weights,
            "budget": redundancy_budget,
            "min_candidate_features": min_candidate_features,
        },
        "ga": {
            "population_size": ga_config.population_size,
            "generations": ga_config.generations,
            "mutation_prob": ga_config.mutation_prob,
            "crossover_prob": ga_config.crossover_prob,
            "elitism": ga_config.elitism,
            "tournament_size": ga_config.tournament_size,
            "min_features": ga_config.min_features,
            "max_features": ga_config.max_features,
            "random_state": ga_config.random_state,
            "lambda_penalty": lambda_penalty,
            "alpha_size": alpha_size,
            "cost_beta": cost_beta,
            "ga_cv_splits": ga_cv_splits,
        },
    }
    config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def get_cache_path(data_path: Path, cache_key: str) -> Path:
    """Get the cache directory path for a given cache key."""
    data_root = ExperimentTracker.compute_data_root(data_path)
    cache_root = data_root / "cache" / "ga_features"
    cache_dir = cache_root / cache_key
    return cache_dir


def save_ga_cache(
    cache_dir: Path,
    data: Dict,
    selected_features: list,
    ga_score: float,
    redundancy_penalty: float,
    penalty_matrix: pd.DataFrame,
    candidate_features: list,
    ensemble_scores: Dict[str, float],
    history: Optional[list] = None,
    constructed_frames: Optional[Dict[str, pd.DataFrame]] = None,
    constructed_metadata: Optional[Dict] = None,
) -> None:
    """Save GA features cache to a fixed location."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected features
    features_path = cache_dir / "selected_features.json"
    features_path.write_text(json.dumps(selected_features), encoding="utf-8")
    
    # Save GA metadata
    ga_meta = {
        "best_score": ga_score,
        "redundancy_penalty": redundancy_penalty,
        "num_selected": len(selected_features),
    }
    if history is not None:
        history_path = cache_dir / "ga_history.json"
        history_path.write_text(json.dumps(history), encoding="utf-8")
        ga_meta["has_history"] = True
    else:
        ga_meta["has_history"] = False
    
    meta_path = cache_dir / "ga_metadata.json"
    meta_path.write_text(json.dumps(ga_meta), encoding="utf-8")
    
    # Save penalty matrix and candidate features
    penalty_path = cache_dir / "penalty_matrix.pkl"
    penalty_matrix.to_pickle(penalty_path)
    
    candidate_path = cache_dir / "candidate_features.json"
    candidate_path.write_text(json.dumps(candidate_features), encoding="utf-8")
    
    # Save ensemble scores
    scores_path = cache_dir / "ensemble_scores.json"
    scores_path.write_text(json.dumps({k: float(v) for k, v in ensemble_scores.items()}), encoding="utf-8")
    
    # Save constructed features if available
    if constructed_frames:
        for key, frame in constructed_frames.items():
            frame_path = cache_dir / f"{key}_constructed.pkl"
            frame.to_pickle(frame_path)
    
    if constructed_metadata:
        metadata_path = cache_dir / "constructed_metadata.json"
        metadata_path.write_text(json.dumps(constructed_metadata), encoding="utf-8")
    
    # Save cache info
    cache_info = {
        "cache_key": cache_dir.name,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    info_path = cache_dir / "cache_info.json"
    info_path.write_text(json.dumps(cache_info), encoding="utf-8")
    
    print(f"[Cache] GA features saved to: {cache_dir}")


def load_ga_cache(cache_dir: Path) -> Dict:
    """Load GA features cache from a fixed location."""
    if not cache_dir.exists():
        return None
    
    # Load selected features
    features_path = cache_dir / "selected_features.json"
    if not features_path.exists():
        return None
    
    selected_features = json.loads(features_path.read_text(encoding="utf-8"))
    
    # Load GA metadata
    meta_path = cache_dir / "ga_metadata.json"
    if meta_path.exists():
        ga_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        ga_meta = {}
    
    # Load history if available
    history = None
    if ga_meta.get("has_history"):
        history_path = cache_dir / "ga_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    
    # Load penalty matrix and candidate features
    penalty_path = cache_dir / "penalty_matrix.pkl"
    penalty_matrix = pd.read_pickle(penalty_path) if penalty_path.exists() else None
    
    candidate_path = cache_dir / "candidate_features.json"
    candidate_features = json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else None
    
    # Load ensemble scores
    scores_path = cache_dir / "ensemble_scores.json"
    ensemble_scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
    
    # Load constructed features if available
    constructed_frames = {}
    constructed_metadata = None
    for key in ["X_train_res", "X_train_scaled", "X_val_scaled", "X_test_scaled"]:
        frame_path = cache_dir / f"{key}_constructed.pkl"
        if frame_path.exists():
            constructed_frames[key] = pd.read_pickle(frame_path)
    
    metadata_path = cache_dir / "constructed_metadata.json"
    if metadata_path.exists():
        constructed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    return {
        "selected_features": selected_features,
        "best_score": ga_meta.get("best_score"),
        "redundancy_penalty": ga_meta.get("redundancy_penalty"),
        "history": history,
        "penalty_matrix": penalty_matrix,
        "candidate_features": candidate_features,
        "ensemble_scores": ensemble_scores,
        "constructed_frames": constructed_frames,
        "constructed_metadata": constructed_metadata,
    }


def run_ga_preparation(args: argparse.Namespace) -> None:
    """Run preprocessing through GA and save to cache."""
    random_state = args.random_state
    
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
    
    ensemble_weights = {
        "pca": args.weight_pca,
        "mutual_info": args.weight_mi,
        "random_forest": args.weight_rf,
    }
    
    penalty_weights = {
        "cmi": args.penalty_weight_cmi,
        "corr": args.penalty_weight_corr,
        "vif": args.penalty_weight_vif,
    }
    
    ga_config = GAConfig(
        population_size=args.ga_population,
        generations=args.ga_generations,
        mutation_prob=args.ga_mutation,
        random_state=random_state,
    )
    
    # Compute cache key
    cache_key = compute_cache_key(
        args.data_path,
        preprocessing_config,
        ensemble_weights,
        args.feature_ensemble_mode,
        args.feature_ensemble_top_k,
        penalty_weights,
        args.redundancy_budget,
        args.min_candidate_features,
        ga_config,
        args.lambda_penalty,
        args.alpha_size,
        args.cost_beta,
        args.ga_cv_splits,
    )
    
    cache_dir = get_cache_path(args.data_path, cache_key)
    
    # Check if cache exists
    if cache_dir.exists() and (cache_dir / "selected_features.json").exists():
        print(f"[Cache] Found existing GA features cache: {cache_dir}")
        print(f"[Cache] Cache key: {cache_key}")
        if not args.force_rebuild:
            print("[Cache] Use --force-rebuild to regenerate the cache")
            return
    
    print(f"[Cache] Generating GA features cache (key: {cache_key})...")
    
    # ------------------------------------------------------------------
    # Stage: Preprocessing
    # ------------------------------------------------------------------
    print("[Cache] Stage 1/4: Preprocessing...")
    data = load_and_preprocess(args.data_path, preprocessing_config)
    feature_columns = list(data["X_train_res"].columns)
    print(f"[Cache] Preprocessing completed: {len(feature_columns)} features")
    
    # ------------------------------------------------------------------
    # Stage: Feature Scoring
    # ------------------------------------------------------------------
    print("[Cache] Stage 2/4: Feature Scoring...")
    
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
    
    constructed_frames: Optional[Dict[str, pd.DataFrame]] = None
    constructed_metadata: Optional[Dict] = None
    if args.feature_ensemble_mode == "construct":
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
            top_k=args.feature_ensemble_top_k,
        )
        if constructed_frames:
            for key, frame in constructed_frames.items():
                data[key] = pd.concat([data[key], frame], axis=1)
        if constructed_metadata:
            new_scores = constructed_metadata.get("new_scores", {})
            if new_scores:
                ensemble_scores.update({k: float(v) for k, v in new_scores.items()})
    
    print(f"[Cache] Feature scoring completed: {len(ensemble_scores)} features scored")
    
    # ------------------------------------------------------------------
    # Stage: Redundancy minimisation
    # ------------------------------------------------------------------
    print("[Cache] Stage 3/4: Redundancy Minimisation...")
    
    penalty_matrix = build_redundancy_penalty_matrix(
        data["X_train_res"],
        data["y_train_res"],
        penalty_weights,
    )
    
    candidate_features = redundancy_aware_selection(
        ensemble_scores,
        penalty_matrix,
        ensemble_scores,
        budget=args.redundancy_budget,
        min_features=args.min_candidate_features,
    )
    
    print(f"[Cache] Redundancy minimisation completed: {len(candidate_features)} candidate features")
    
    # ------------------------------------------------------------------
    # Stage: Genetic Algorithm
    # ------------------------------------------------------------------
    print("[Cache] Stage 4/4: Genetic Algorithm...")
    
    fitness_fn = make_cost_sensitive_fitness(
        feature_names=candidate_features,
        ensemble_weights=ensemble_weights,
        penalty_matrix=penalty_matrix,
        lambda_penalty=args.lambda_penalty,
        alpha_size=args.alpha_size,
        cost_beta=args.cost_beta,
        random_state=random_state,
        cv_splits=args.ga_cv_splits,
    )
    
    estimator = LogisticRegression(
        max_iter=1500,
        solver="lbfgs",
        class_weight=None,
        random_state=random_state,
    )
    
    ga_config.min_features = min(args.ga_min_features, len(candidate_features))
    ga_config.max_features = len(candidate_features)
    
    selector = GeneticFeatureSelector(
        estimator=estimator,
        config=ga_config,
        verbose=not args.ga_quiet,
        fitness_function=fitness_fn,
    )
    
    selector.fit(data["X_train_res"][candidate_features], data["y_train_res"])
    selected_features = selector.get_feature_names()
    ga_score = selector.best_score_
    redundancy_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_scores)
    
    print(f"[Cache] GA completed: {len(selected_features)} features selected (score: {ga_score:.4f})")
    
    # Save to cache
    save_ga_cache(
        cache_dir,
        data,
        selected_features,
        ga_score,
        redundancy_penalty,
        penalty_matrix,
        candidate_features,
        ensemble_scores,
        history=selector.history_,
        constructed_frames=constructed_frames,
        constructed_metadata=constructed_metadata,
    )
    
    print(f"[Cache] GA features cache saved successfully!")
    print(f"[Cache] Cache location: {cache_dir}")
    print(f"[Cache] This cache will be automatically used by main.py and main_commercial.py")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (same as main.py for consistency)."""
    parser = argparse.ArgumentParser(
        description="Prepare GA features cache: Preprocessing → Feature Scoring → Redundancy → GA"
    )
    
    # Data arguments
    parser.add_argument("--data-path", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--target-column", type=str, default="Class", help="Target column name")
    
    # Preprocessing arguments
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--enable-undersampling", action="store_true", help="Enable undersampling")
    parser.add_argument("--undersample-ratio", type=float, default=None, help="Undersampling ratio")
    parser.add_argument("--cost-beta", type=float, default=5.0, help="Cost-sensitive weight for fraud class")
    
    # Feature scoring arguments
    parser.add_argument("--weight-pca", type=float, default=0.3, help="PCA weight in ensemble")
    parser.add_argument("--weight-mi", type=float, default=0.3, help="Mutual information weight in ensemble")
    parser.add_argument("--weight-rf", type=float, default=0.4, help="Random forest weight in ensemble")
    parser.add_argument(
        "--feature-ensemble-mode",
        type=str,
        default="scores",
        choices=["scores", "construct"],
        help="Ensemble mode: 'scores' or 'construct'",
    )
    parser.add_argument(
        "--feature-ensemble-top-k",
        type=int,
        default=10,
        help="Top-k features per method for construction mode",
    )
    
    # Redundancy arguments
    parser.add_argument("--penalty-weight-cmi", type=float, default=0.4, help="CMI penalty weight")
    parser.add_argument("--penalty-weight-corr", type=float, default=0.3, help="Correlation penalty weight")
    parser.add_argument("--penalty-weight-vif", type=float, default=0.3, help="VIF penalty weight")
    parser.add_argument("--redundancy-budget", type=float, default=0.8, help="Redundancy budget")
    parser.add_argument("--min-candidate-features", type=int, default=10, help="Minimum candidate features")
    
    # GA arguments
    parser.add_argument("--ga-population", type=int, default=40, help="GA population size")
    parser.add_argument("--ga-generations", type=int, default=40, help="GA generations")
    parser.add_argument("--ga-mutation", type=float, default=0.05, help="GA mutation probability")
    parser.add_argument("--ga-min-features", type=int, default=3, help="Minimum features in GA")
    parser.add_argument("--ga-quiet", action="store_true", help="Suppress GA progress output")
    parser.add_argument("--ga-cv-splits", type=int, default=5, help="Cross-validation splits for GA")
    
    # Penalty arguments
    parser.add_argument("--lambda-penalty", type=float, default=0.05, help="Redundancy penalty weight")
    parser.add_argument("--alpha-size", type=float, default=0.01, help="Feature size penalty weight")
    
    # Cache control
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if cache exists",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ga_preparation(args)


if __name__ == "__main__":
    main()

