"""GA integration and execution utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Module import ExperimentTracker, GAConfig, GeneticFeatureSelector
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
)
from utils.feature_selection_utils import compute_subset_penalty
from utils.optimization_utils import make_cost_sensitive_fitness
from utils.transfer_learning import load_previous_best_solutions


def load_ga_from_cache(
    tracker: ExperimentTracker,
    data_path: Path,
    preprocessing_config,
    ensemble_weights: Dict[str, float],
    feature_ensemble_mode: str,
    feature_ensemble_top_k: int,
    penalty_weights: Dict[str, float],
    redundancy_budget: float,
    min_candidate_features: int,
    ga_config: GAConfig,
    lambda_penalty: float,
    alpha_size: float,
    cost_beta: float,
    ga_cv_splits: int,
    random_state: int,
) -> Optional[Dict]:
    """Try to load GA results from fixed cache.
    
    Returns
    -------
    Dictionary with GA results if found, None otherwise.
    """
    try:
        from prepare_ga_features import compute_cache_key, get_cache_path, load_ga_cache
        
        cache_key = compute_cache_key(
            data_path,
            preprocessing_config,
            ensemble_weights,
            feature_ensemble_mode,
            feature_ensemble_top_k,
            penalty_weights,
            redundancy_budget,
            min_candidate_features,
            ga_config,
            lambda_penalty,
            alpha_size,
            cost_beta,
            ga_cv_splits,
        )
        ga_cache_dir = get_cache_path(data_path, cache_key)
        cache_data = load_ga_cache(ga_cache_dir)
        
        if cache_data:
            tracker.log_event("ga", "Loading GA results from fixed cache", {"cache_key": cache_key})
            return {
                "selected_features": cache_data["selected_features"],
                "best_score": cache_data.get("best_score"),
                "redundancy_penalty": cache_data.get("redundancy_penalty"),
                "penalty_matrix": cache_data.get("penalty_matrix"),
                "candidate_features": cache_data.get("candidate_features"),
                "ensemble_scores": cache_data.get("ensemble_scores"),
                "history": cache_data.get("history"),
                "cache_key": cache_key,
            }
    except Exception as e:
        tracker.log_event(
            "ga",
            "Failed to load GA cache, falling back to normal logic",
            {"error": str(e)},
            level=logging.WARNING,
        )
    return None


def load_ga_from_reuse(
    tracker: ExperimentTracker,
    reuse_ga_run: str,
    method_label: str,
    data_path: Path,
    feature_columns: List[str],
) -> Optional[Dict]:
    """Load GA results from a previous run for reuse.
    
    Returns
    -------
    Dictionary with GA results if found, None otherwise.
    """
    print(f"[GA Reuse] Attempting to load GA results from run: {reuse_ga_run}")
    candidate_methods = [method_label]
    if method_label == "with_hessian":
        candidate_methods.append("without_hessian")
    else:
        candidate_methods.append("with_hessian")

    base_root = ExperimentTracker.compute_data_root(data_path)
    print(f"[GA Reuse] Searching in data root: {base_root}")
    print(f"[GA Reuse] Candidate method labels: {candidate_methods}")
    reuse_state_path = None
    reuse_method = None
    for candidate in candidate_methods:
        candidate_path = base_root / "log" / candidate / reuse_ga_run / "state.json"
        print(f"[GA Reuse] Checking: {candidate_path} (exists: {candidate_path.exists()})")
        if candidate_path.exists():
            reuse_state_path = candidate_path
            reuse_method = candidate
            print(f"[GA Reuse] Found state file at: {reuse_state_path}")
            break
    if reuse_state_path is None:
        raise FileNotFoundError(
            f"Reusable GA state not found for any candidate method label. Checked: {', '.join(candidate_methods)}\n"
            f"Searched in: {base_root / 'log'}"
        )
    reuse_state = json.loads(reuse_state_path.read_text(encoding="utf-8"))
    if "ga" not in reuse_state.get("stages", {}):
        raise RuntimeError("Specified reusable GA run does not contain GA stage information.")
    reuse_result_dir = Path(reuse_state["result_dir"])
    print(f"[GA Reuse] Loading GA results from: {reuse_result_dir}")
    ga_stage = reuse_state["stages"]["ga"]
    ga_artifacts = ga_stage["artifacts"]
    selected_features_path = reuse_result_dir / ga_artifacts["selected_features"]["path"]
    if not selected_features_path.exists():
        raise FileNotFoundError(f"GA selected_features file not found: {selected_features_path}")
    selected_features = json.loads(selected_features_path.read_text(encoding="utf-8"))
    print(f"[GA Reuse] Loaded {len(selected_features)} selected features")
    redundancy_penalty = ga_stage["metadata"].get("redundancy_penalty")
    best_score = ga_stage["metadata"].get("best_score")
    history = None
    if "ga_history" in ga_artifacts:
        history_path = reuse_result_dir / ga_artifacts["ga_history"]["path"]
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    return {
        "selected_features": selected_features,
        "redundancy_penalty": redundancy_penalty,
        "best_score": best_score,
        "history": history,
        "source_run": reuse_ga_run,
        "source_method": reuse_method,
    }


def run_ga_stage(
    tracker: ExperimentTracker,
    data: Dict[str, pd.DataFrame],
    candidate_features: List[str],
    ensemble_scores: Dict[str, float],
    penalty_matrix: pd.DataFrame,
    ensemble_weights: Dict[str, float],
    preprocessing_config,
    args,
    random_state: int,
    reuse_ga_data: Optional[Dict] = None,
) -> Tuple[List[str], Optional[float], Optional[float], Optional[List[float]], Optional[List[float]]]:
    """Run genetic algorithm stage.
    
    Parameters
    ----------
    tracker:
        Experiment tracker instance.
    data:
        Dictionary containing preprocessed dataframes.
    candidate_features:
        List of candidate feature names.
    ensemble_scores:
        Dictionary of ensemble feature scores.
    penalty_matrix:
        Redundancy penalty matrix.
    ensemble_weights:
        Dictionary of ensemble weights.
    preprocessing_config:
        Preprocessing configuration object.
    args:
        Command-line arguments namespace.
    random_state:
        Random seed.
    
    Returns
    -------
    Tuple of (selected_features, ga_score, redundancy_penalty, ga_history, diversity_history).
    """
    # Try cache first - use the same config type that will be used for GA
    ga_cache_result = None
    try:
        # Build the same config that will be used for GA execution
        if args.use_advanced_ga:
            cache_ga_config = AdvancedGAConfig(
                population_size=args.ga_population,
                generations=args.ga_generations,
                min_crossover_prob=getattr(args, 'ga_min_crossover_prob', 0.5),
                max_crossover_prob=getattr(args, 'ga_max_crossover_prob', 0.95),
                min_mutation_prob=args.ga_mutation * 0.5,
                max_mutation_prob=args.ga_mutation * 2.0,
                mutation_sigma_init=getattr(args, 'ga_mutation_sigma_init', 0.1),
                mutation_sigma_min=getattr(args, 'ga_mutation_sigma_min', 0.01),
                mutation_sigma_max=getattr(args, 'ga_mutation_sigma_max', 1.0),
                random_state=random_state,
                crossover_type=args.ga_crossover_type,
                use_local_search=args.ga_local_search,
                local_search_type=args.ga_local_search_type,
                sa_initial_temp=getattr(args, 'sa_initial_temp', 100.0),
                sa_cooling_rate=getattr(args, 'sa_cooling_rate', 0.95),
                use_fitness_sharing=args.ga_fitness_sharing,
                fitness_sharing_sigma=args.ga_fitness_sharing_sigma,
                fitness_sharing_alpha=getattr(args, 'ga_fitness_sharing_alpha', 1.0),
                use_surrogate=args.ga_surrogate,
                surrogate_type=args.ga_surrogate_type,
                surrogate_update_interval=getattr(args, 'ga_surrogate_update_interval', 5),
                surrogate_sample_size=getattr(args, 'ga_surrogate_sample_size', 100),
                adaptive_population=args.ga_adaptive_population,
                population_alpha=getattr(args, 'ga_population_alpha', 0.5),
                population_beta=getattr(args, 'ga_population_beta', 1.5),
                early_stopping_patience=args.ga_early_stopping,
                heuristic_init_ratio=args.ga_heuristic_init,
                use_island_model=getattr(args, 'ga_island_model', False),
                num_islands=getattr(args, 'ga_num_islands', 4),
                migration_interval=getattr(args, 'ga_migration_interval', 10),
                migration_rate=getattr(args, 'ga_migration_rate', 0.1),
                use_multi_objective=getattr(args, 'ga_multi_objective', False),
                replacement_strategy=getattr(args, 'ga_replacement_strategy', 'generational'),
                mu_plus_lambda_mu=getattr(args, 'ga_mu_plus_lambda_mu', None),
                steady_state_replace_worst=True,
                use_tabu_search=getattr(args, 'ga_tabu_search', False),
                tabu_tenure=getattr(args, 'ga_tabu_tenure', 5),
                use_transfer_learning=args.ga_transfer_learning,
            )
        elif args.use_enhanced_ga:
            cache_ga_config = EnhancedGAConfig(
                population_size=args.ga_population,
                generations=args.ga_generations,
                mutation_prob=args.ga_mutation,
                random_state=random_state,
                crossover_type=args.ga_crossover_type,
                use_local_search=args.ga_local_search,
                early_stopping_patience=args.ga_early_stopping,
                heuristic_init_ratio=args.ga_heuristic_init,
            )
        else:
            cache_ga_config = GAConfig(
                population_size=args.ga_population,
                generations=args.ga_generations,
                mutation_prob=args.ga_mutation,
                random_state=random_state,
            )
        
        ga_cache_result = load_ga_from_cache(
            tracker,
            args.data_path,
            preprocessing_config,
            ensemble_weights,
            args.feature_ensemble_mode,
            args.feature_ensemble_top_k,
            {
                "cmi": args.penalty_weight_cmi,
                "corr": args.penalty_weight_corr,
                "vif": args.penalty_weight_vif,
            },
            args.redundancy_budget,
            args.min_candidate_features,
            cache_ga_config,
            args.lambda_penalty,
            args.alpha_size,
            args.cost_beta,
            args.ga_cv_splits,
            random_state,
        )
    except Exception:
        pass
    
    if ga_cache_result:
        selected_features = ga_cache_result["selected_features"]
        ga_score = ga_cache_result.get("best_score")
        redundancy_penalty = ga_cache_result.get("redundancy_penalty")
        ga_history = ga_cache_result.get("history")
        diversity_history = None
        
        # Update penalty_matrix and candidate_features from cache if available
        if ga_cache_result.get("penalty_matrix") is not None:
            penalty_matrix = ga_cache_result["penalty_matrix"]
        if ga_cache_result.get("candidate_features") is not None:
            candidate_features = ga_cache_result["candidate_features"]
        if ga_cache_result.get("ensemble_scores"):
            ensemble_scores.update(ga_cache_result["ensemble_scores"])
        
        # Save to tracker for consistency
        tracker.save_ga_results(
            selected_features,
            ga_score,
            history=ga_history,
            redundancy_penalty=redundancy_penalty,
        )
        
        # Plot GA convergence from cache
        if args.plot_convergence and ga_history:
            try:
                ga_plot_path = plot_ga_convergence(
                    ga_history,
                    tracker.result_dir / "ga" / "convergence.png",
                    title="GA Convergence (from cache)",
                )
                tracker.log_event(
                    "ga",
                    "Generated convergence plot from cache",
                    {"plot_path": str(ga_plot_path.relative_to(tracker.result_dir))},
                )
                print(f"[GA] Convergence plot saved to: {ga_plot_path}")
            except Exception as e:
                tracker.log_event(
                    "ga",
                    "Failed to generate convergence plot from cache",
                    {"error": str(e)},
                    level=logging.WARNING,
                )
        
        tracker.log_event(
            "ga",
            "GA results loaded from cache",
            {
                "cache_key": ga_cache_result.get("cache_key"),
                "selected_features": len(selected_features),
            },
        )
        return selected_features, ga_score, redundancy_penalty, ga_history, diversity_history
    
    # Check tracker
    if tracker.is_completed("ga"):
        tracker.log_event("ga", "Loading cached GA results")
        ga_data = tracker.load_ga_results()
        selected_features = ga_data["selected_features"]
        ga_score = ga_data.get("best_score")
        redundancy_penalty = ga_data.get("redundancy_penalty")
        ga_history = ga_data.get("history")
        diversity_history = None
        
        # Plot GA convergence from saved results
        if args.plot_convergence and ga_history:
            try:
                ga_plot_path = plot_ga_convergence(
                    ga_history,
                    tracker.result_dir / "ga" / "convergence.png",
                    title="GA Convergence (from saved results)",
                )
                tracker.log_event(
                    "ga",
                    "Generated convergence plot from saved results",
                    {"plot_path": str(ga_plot_path.relative_to(tracker.result_dir))},
                )
                print(f"[GA] Convergence plot saved to: {ga_plot_path}")
            except Exception as e:
                tracker.log_event(
                    "ga",
                    "Failed to generate convergence plot from saved results",
                    {"error": str(e)},
                    level=logging.WARNING,
                )
        return selected_features, ga_score, redundancy_penalty, ga_history, diversity_history
    
    # Check reuse
    if reuse_ga_data is not None:
        print(f"[GA] Reusing GA results from run: {reuse_ga_data['source_run']} (method: {reuse_ga_data['source_method']})")
        feature_columns = list(data["X_train_res"].columns)
        selected_features = [feat for feat in reuse_ga_data["selected_features"] if feat in feature_columns]
        if not selected_features:
            raise RuntimeError("Reusable GA features are not present in current dataset.")
        redundancy_penalty = reuse_ga_data["redundancy_penalty"]
        if redundancy_penalty is None:
            redundancy_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_scores)
        ga_score = reuse_ga_data.get("best_score")
        tracker.save_ga_results(
            selected_features,
            ga_score,
            history=reuse_ga_data.get("history"),
            redundancy_penalty=redundancy_penalty,
        )
        tracker.log_event(
            "ga",
            "Reused GA results",
            {
                "source_run": reuse_ga_data["source_run"],
                "source_method": reuse_ga_data["source_method"],
                "selected_features": len(selected_features),
            },
        )
        return selected_features, ga_score, redundancy_penalty, reuse_ga_data.get("history"), None
    
    # Run GA
    tracker.log_event("ga", "Starting genetic algorithm optimisation")
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

    # Use advanced GA if enabled (highest level)
    if args.use_advanced_ga:
        # Transfer learning: try to load previous best solutions
        transfer_solutions = None
        if args.ga_transfer_learning:
            transfer_solutions = load_previous_best_solutions(
                args.data_path, tracker.method_label, candidate_features
            )
        
        ga_config = AdvancedGAConfig(
            population_size=args.ga_population,
            generations=args.ga_generations,
            min_mutation_prob=args.ga_mutation * 0.5,
            max_mutation_prob=args.ga_mutation * 2.0,
            min_features=min(args.ga_min_features, len(candidate_features)),
            max_features=len(candidate_features),
            random_state=random_state,
            crossover_type=args.ga_crossover_type,
            use_local_search=args.ga_local_search,
            local_search_type=args.ga_local_search_type,
            use_fitness_sharing=args.ga_fitness_sharing,
            fitness_sharing_sigma=args.ga_fitness_sharing_sigma,
            use_surrogate=args.ga_surrogate,
            surrogate_type=args.ga_surrogate_type,
            adaptive_population=args.ga_adaptive_population,
            early_stopping_patience=args.ga_early_stopping,
            heuristic_init_ratio=args.ga_heuristic_init,
            use_island_model=args.ga_island_model,
            num_islands=args.ga_num_islands,
            migration_interval=args.ga_migration_interval,
            migration_rate=args.ga_migration_rate,
            use_multi_objective=args.ga_multi_objective,
            replacement_strategy=args.ga_replacement_strategy,
            mu_plus_lambda_mu=args.ga_mu_plus_lambda_mu,
            use_tabu_search=args.ga_tabu_search,
            tabu_tenure=args.ga_tabu_tenure,
            use_transfer_learning=args.ga_transfer_learning,
            transfer_solutions=transfer_solutions,
        )

        selector = AdvancedGeneticFeatureSelector(
            estimator=estimator,
            config=ga_config,
            verbose=not args.ga_quiet,
            fitness_function=fitness_fn,
            feature_importance=ensemble_scores,
        )
    # Use enhanced GA if enabled
    elif args.use_enhanced_ga:
        ga_config = EnhancedGAConfig(
            population_size=args.ga_population,
            generations=args.ga_generations,
            mutation_prob=args.ga_mutation,
            min_features=min(args.ga_min_features, len(candidate_features)),
            max_features=len(candidate_features),
            random_state=random_state,
            crossover_type=args.ga_crossover_type,
            use_local_search=args.ga_local_search,
            early_stopping_patience=args.ga_early_stopping,
            heuristic_init_ratio=args.ga_heuristic_init,
        )

        selector = EnhancedGeneticFeatureSelector(
            estimator=estimator,
            config=ga_config,
            verbose=not args.ga_quiet,
            fitness_function=fitness_fn,
            feature_importance=ensemble_scores,
        )
    else:
        ga_config = GAConfig(
            population_size=args.ga_population,
            generations=args.ga_generations,
            mutation_prob=args.ga_mutation,
            min_features=min(args.ga_min_features, len(candidate_features)),
            max_features=len(candidate_features),
            random_state=random_state,
        )

        selector = GeneticFeatureSelector(
            estimator=estimator,
            config=ga_config,
            verbose=not args.ga_quiet,
            fitness_function=fitness_fn,
        )

    selector.fit(data["X_train_res"][candidate_features], data["y_train_res"])
    selected_features = selector.get_feature_names()
    ga_score = selector.best_score_
    redundancy_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_weights)
    
    # Save GA results
    ga_history = selector.history_
    diversity_history = None
    if (args.use_advanced_ga or args.use_enhanced_ga) and hasattr(selector, 'diversity_history_'):
        diversity_history = selector.diversity_history_
    
    tracker.save_ga_results(
        selected_features,
        ga_score,
        history=ga_history,
        redundancy_penalty=redundancy_penalty,
    )
    
    # Plot GA convergence
    if args.plot_convergence and ga_history:
        try:
            if (args.use_advanced_ga or args.use_enhanced_ga) and diversity_history:
                title = "Advanced GA Convergence" if args.use_advanced_ga else "Enhanced GA Convergence"
                ga_plot_path = plot_enhanced_ga_convergence(
                    ga_history,
                    diversity_history,
                    tracker.result_dir / "ga" / "convergence.png",
                    title=title,
                )
            else:
                ga_plot_path = plot_ga_convergence(
                    ga_history,
                    tracker.result_dir / "ga" / "convergence.png",
                    title="GA Convergence",
                )
            tracker.log_event(
                "ga",
                "Generated convergence plot",
                {"plot_path": str(ga_plot_path.relative_to(tracker.result_dir))},
            )
            print(f"[GA] Convergence plot saved to: {ga_plot_path}")
        except Exception as e:
            tracker.log_event(
                "ga",
                "Failed to generate convergence plot",
                {"error": str(e)},
                level=logging.WARNING,
            )
    
    tracker.log_event(
        "ga",
        "Genetic algorithm completed",
        {
            "selected_features": len(selected_features),
            "best_score": ga_score,
            "redundancy_penalty": redundancy_penalty,
            "enhanced": args.use_enhanced_ga,
            "advanced": args.use_advanced_ga,
        },
    )
    
    return selected_features, ga_score, redundancy_penalty, ga_history, diversity_history

