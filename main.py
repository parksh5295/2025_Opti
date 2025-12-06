"""Information-theoretic ensemble pipeline for credit-card fraud detection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Module import (
    EvaluationConfig,
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
from utils.convergence_plots import (
    plot_combined_convergence,
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
from utils.argument_parsers import create_main_parser
from utils.optimization_utils import (
    cost_sensitive_negative_log_likelihood,
    make_cost_sensitive_fitness,
    optimise_threshold,
)


def run_pipeline(args: argparse.Namespace) -> None:
    random_state = args.random_state
    use_second_order = (
        args.solver_backend == "custom" and args.solver_second_order == "bfgs"
    )
    method_label = "with_hessian" if use_second_order else "without_hessian"

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

    reuse_ga_data = None
    if args.reuse_ga_run:
        print(f"[GA Reuse] Attempting to load GA results from run: {args.reuse_ga_run}")
        if args.reuse_ga_method:
            candidate_methods = [args.reuse_ga_method]
        else:
            candidate_methods = [method_label]
            if method_label == "with_hessian":
                candidate_methods.append("without_hessian")
            else:
                candidate_methods.append("with_hessian")

        base_root = ExperimentTracker.compute_data_root(args.data_path)
        print(f"[GA Reuse] Searching in data root: {base_root}")
        print(f"[GA Reuse] Candidate method labels: {candidate_methods}")
        reuse_state_path = None
        reuse_method = None
        for candidate in candidate_methods:
            candidate_path = base_root / "log" / candidate / args.reuse_ga_run / "state.json"
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
        reuse_ga_data = {
            "selected_features": selected_features,
            "redundancy_penalty": redundancy_penalty,
            "best_score": best_score,
            "history": history,
            "source_run": args.reuse_ga_run,
            "source_method": reuse_method,
        }
        print(f"[GA Reuse] Successfully loaded GA data from run: {args.reuse_ga_run} (method: {reuse_method})")

    try:
        # ------------------------------------------------------------------
        # Stage: Preprocessing
        # ------------------------------------------------------------------
        if tracker.is_completed("preprocessing"):
            tracker.log_event("preprocessing", "Loading cached preprocessing outputs")
            data = tracker.load_preprocessing()
        else:
            tracker.log_event("preprocessing", "Starting preprocessing")
            data = load_and_preprocess(args.data_path, preprocessing_config)
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

        feature_columns = list(data["X_train_res"].columns)

        # ------------------------------------------------------------------
        # Stage: Feature Scoring
        # ------------------------------------------------------------------
        ensemble_weights = {
            "pca": args.weight_pca,
            "mutual_info": args.weight_mi,
            "random_forest": args.weight_rf,
        }
        constructed_frames: Optional[Dict[str, pd.DataFrame]] = None
        constructed_metadata: Optional[Dict[str, object]] = None
        if tracker.is_completed("feature_scoring"):
            tracker.log_event("feature_scoring", "Loading cached feature scores")
            scores_payload = tracker.load_feature_scores()
            score_map = scores_payload["scores"]
            pca_scores = score_map.get("pca", {})
            mi_scores = score_map.get("mi", {})
            rf_scores = score_map.get("rf", {})
            ensemble_scores = score_map.get("ensemble", {})

            constructed_frames = scores_payload.get("constructed_frames") or {}
            metadata_fs = scores_payload.get("metadata", {})
            stored_mode = metadata_fs.get("ensemble_mode")
            if stored_mode and stored_mode != args.feature_ensemble_mode:
                tracker.log_event(
                    "feature_scoring",
                    "Stored ensemble mode differs from requested mode; continuing with stored configuration.",
                    {"stored_mode": stored_mode, "requested_mode": args.feature_ensemble_mode},
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
                ensemble_mode=args.feature_ensemble_mode,
                constructed_frames=constructed_frames or None,
                constructed_metadata=constructed_metadata,
            )
            tracker.log_event(
                "feature_scoring",
                "Feature scoring completed",
                {"num_features": len(ensemble_scores)},
            )

        # ------------------------------------------------------------------
        # Stage: Redundancy minimisation
        # ------------------------------------------------------------------
        if tracker.is_completed("redundancy"):
            tracker.log_event("redundancy", "Loading cached redundancy analysis")
            redundancy_data = tracker.load_redundancy()
            penalty_matrix = redundancy_data["penalty_matrix"]
            candidate_features = redundancy_data["candidate_features"]
        else:
            tracker.log_event("redundancy", "Computing redundancy penalty matrix")
            penalty_weights = {
                "cmi": args.penalty_weight_cmi,
                "corr": args.penalty_weight_corr,
                "vif": args.penalty_weight_vif,
            }
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
            tracker.save_redundancy(penalty_matrix, candidate_features)
            tracker.log_event(
                "redundancy",
                "Candidate feature set generated",
                {"num_candidates": len(candidate_features)},
            )

        print(
            f"[Pipeline] Candidate features after redundancy-aware selection: {len(candidate_features)}"
        )

        # ------------------------------------------------------------------
        # Stage: Genetic Algorithm
        # ------------------------------------------------------------------
        # Try to load from GA cache first (fixed name, not date-based)
        ga_cache_dir = None
        try:
            from prepare_ga_features import compute_cache_key, get_cache_path, load_ga_cache
            
            cache_key = compute_cache_key(
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
                GAConfig(
                    population_size=args.ga_population,
                    generations=args.ga_generations,
                    mutation_prob=args.ga_mutation,
                    random_state=random_state,
                ),
                args.lambda_penalty,
                args.alpha_size,
                args.cost_beta,
                args.ga_cv_splits,
            )
            ga_cache_dir = get_cache_path(args.data_path, cache_key)
            cache_data = load_ga_cache(ga_cache_dir)
            
            if cache_data:
                tracker.log_event("ga", "Loading GA results from fixed cache", {"cache_key": cache_key})
                selected_features = cache_data["selected_features"]
                ga_score = cache_data.get("best_score")
                redundancy_penalty = cache_data.get("redundancy_penalty")
                # Update penalty_matrix and candidate_features from cache if available
                if cache_data.get("penalty_matrix") is not None:
                    penalty_matrix = cache_data["penalty_matrix"]
                if cache_data.get("candidate_features") is not None:
                    candidate_features = cache_data["candidate_features"]
                if cache_data.get("ensemble_scores"):
                    ensemble_scores.update(cache_data["ensemble_scores"])
                
                # Load history for plotting
                ga_history = cache_data.get("history")
                diversity_history = None  # Cache doesn't store diversity history yet
                
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
                        "cache_key": cache_key,
                        "selected_features": len(selected_features),
                    },
                )
            else:
                ga_cache_dir = None  # Cache not found, fall through to normal logic
        except Exception as e:
            tracker.log_event(
                "ga",
                "Failed to load GA cache, falling back to normal logic",
                {"error": str(e)},
                level=logging.WARNING,
            )
            ga_cache_dir = None
        
        if ga_cache_dir is None:
            # Normal logic: check tracker or reuse_ga_data
            if tracker.is_completed("ga"):
                tracker.log_event("ga", "Loading cached GA results")
                ga_data = tracker.load_ga_results()
                selected_features = ga_data["selected_features"]
                ga_score = ga_data.get("best_score")
                redundancy_penalty = ga_data.get("redundancy_penalty")
                ga_history = ga_data.get("history")
                
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
            elif reuse_ga_data is not None:
                print(f"[GA] Reusing GA results from run: {reuse_ga_data['source_run']} (method: {reuse_ga_data['source_method']})")
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
            else:
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

                # Use enhanced GA if enabled
                if args.use_enhanced_ga:
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
                if args.use_enhanced_ga and hasattr(selector, 'diversity_history_'):
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
                        if args.use_enhanced_ga and diversity_history:
                            ga_plot_path = plot_enhanced_ga_convergence(
                                ga_history,
                                diversity_history,
                                tracker.result_dir / "ga" / "convergence.png",
                                title="Enhanced GA Convergence",
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
                    },
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
        weights: Optional[np.ndarray] = None
        bias: Optional[float] = None
        snapshots: Optional[List[Dict[str, object]]] = None
        threshold: float
        val_score: float

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

            if args.solver_backend == "custom":
                val_weights = compute_sample_weights(data["y_val"], args.cost_beta)
                threshold, val_score = optimise_threshold(
                    data["y_val"],
                    val_probabilities,
                    beta=args.beta,
                    sample_weight=val_weights,
                )
            else:
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

        print(f"[Solver] Optimal decision threshold: {threshold:.3f} (F{args.beta:.1f}={val_score:.4f})")

        # ------------------------------------------------------------------
        # Stage: Evaluation
        # ------------------------------------------------------------------
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

        if not args.skip_explainability:
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

