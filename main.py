"""Information-theoretic ensemble pipeline for credit-card fraud detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

from Module import (
    EvaluationConfig,
    ExperimentTracker,
    GAConfig,
    GeneticFeatureSelector,
    PreprocessingConfig,
    SolverConfig,
    compute_sample_weights,
    configure_sklearn_like_model,
    evaluate_model,
    format_cost_sensitive_summary,
    load_and_preprocess,
    solve_cost_sensitive_logistic,
    solver_predict_proba,
)


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val < 1e-9:
        return {feat: 1.0 for feat in scores}
    return {feat: (val - min_val) / (max_val - min_val) for feat, val in scores.items()}


def compute_pca_scores(
    X_train: pd.DataFrame,
    variance_threshold: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    pca.fit(X_train)

    loadings = np.abs(pca.components_) * pca.explained_variance_ratio_[:, np.newaxis]
    scores = loadings.sum(axis=0)
    score_map = dict(zip(X_train.columns, scores))
    return normalise_scores(score_map)


def compute_mutual_information_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, float]:
    mi = mutual_info_classif(
        X_train,
        y_train,
        random_state=random_state,
    )
    score_map = dict(zip(X_train.columns, mi))
    return normalise_scores(score_map)


def compute_random_forest_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, float]:
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    score_map = dict(zip(X_train.columns, importances))
    return normalise_scores(score_map)


def information_theoretic_ensemble_scores(
    scores: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for source, feature_scores in scores.items():
        weight = weights.get(source, 0.0)
        for feature, value in feature_scores.items():
            combined.setdefault(feature, 0.0)
            combined[feature] += weight * value
    return combined


def compute_conditional_mutual_information_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    discretiser = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_disc = discretiser.fit_transform(X)
    feature_names = X.columns
    classes = np.unique(y)
    cmi_matrix = np.zeros((len(feature_names), len(feature_names)), dtype=float)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            value = 0.0
            for cls in classes:
                mask = y.values == cls
                if np.sum(mask) < 2:
                    continue
                mi = mutual_info_score(X_disc[mask, i], X_disc[mask, j])
                value += (np.sum(mask) / len(y)) * mi
            cmi_matrix[i, j] = cmi_matrix[j, i] = value

    return pd.DataFrame(cmi_matrix, index=feature_names, columns=feature_names).fillna(0.0)


def compute_vif_scores(X: pd.DataFrame) -> pd.Series:
    values = X.values
    n_features = values.shape[1]
    vif_scores = []

    for i in range(n_features):
        y_col = values[:, i]
        X_other = np.delete(values, i, axis=1)
        X_other = np.column_stack([np.ones(len(X_other)), X_other])
        coef, _, _, _ = np.linalg.lstsq(X_other, y_col, rcond=None)
        y_pred = X_other @ coef
        residuals = y_col - y_pred
        sse = np.sum(residuals ** 2)
        sst = np.sum((y_col - y_col.mean()) ** 2)
        if sst <= 0:
            vif = 1.0
        else:
            r_squared = 1.0 - (sse / sst)
            vif = 1.0 / max(1.0 - r_squared, 1e-6)
        vif_scores.append(vif)

    return pd.Series(vif_scores, index=X.columns, name="vif")


def _normalise_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    max_val = matrix.values.max()
    min_val = matrix.values.min()
    if max_val - min_val < 1e-9:
        return pd.DataFrame(1.0, index=matrix.index, columns=matrix.columns)
    norm_values = (matrix - min_val) / (max_val - min_val)
    return norm_values.fillna(0.0)


def build_redundancy_penalty_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    penalty_weights: Dict[str, float],
) -> pd.DataFrame:
    cmi = compute_conditional_mutual_information_matrix(X, y)
    corr = X.corr().abs().fillna(0.0)
    vif = compute_vif_scores(X)
    vif_norm = (vif - vif.min()) / (vif.max() - vif.min() + 1e-9)
    vif_matrix = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
    for i in X.columns:
        for j in X.columns:
            vif_matrix.loc[i, j] = 0.5 * (vif_norm.loc[i] + vif_norm.loc[j])

    combined = (
        penalty_weights.get("cmi", 0.5) * _normalise_matrix(cmi)
        + penalty_weights.get("corr", 0.3) * corr
        + penalty_weights.get("vif", 0.2) * vif_matrix
    )

    return combined.fillna(0.0)


def redundancy_aware_selection(
    feature_scores: Dict[str, float],
    penalty_matrix: pd.DataFrame,
    ensemble_weights: Dict[str, float],
    budget: float,
    min_features: int,
) -> List[str]:
    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    selected: List[str] = []

    for feature, _ in ranked:
        if not selected:
            selected.append(feature)
            continue

        penalty = 0.0
        for chosen in selected:
            penalty += (
                ensemble_weights.get(feature, 0.0)
                * ensemble_weights.get(chosen, 0.0)
                * penalty_matrix.loc[feature, chosen]
            )

        if penalty <= budget or len(selected) < min_features:
            selected.append(feature)

    return selected


def compute_subset_penalty(
    features: Sequence[str],
    penalty_matrix: pd.DataFrame,
    ensemble_weights: Dict[str, float],
) -> float:
    penalty = 0.0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            fi, fj = features[i], features[j]
            penalty += (
                ensemble_weights.get(fi, 0.0)
                * ensemble_weights.get(fj, 0.0)
                * penalty_matrix.loc[fi, fj]
            )
    return penalty


def cost_sensitive_negative_log_likelihood(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sample_weight: ArrayLike,
) -> float:
    eps = 1e-9
    y_prob = np.clip(y_prob, eps, 1 - eps)
    y_true = np.asarray(y_true)
    weights = np.asarray(sample_weight)
    nll = -np.sum(
        weights
        * (
            y_true * np.log(y_prob)
            + (1 - y_true) * np.log(1 - y_prob)
        )
    )
    return nll / np.sum(weights)


def make_cost_sensitive_fitness(
    feature_names: Sequence[str],
    ensemble_weights: Dict[str, float],
    penalty_matrix: pd.DataFrame,
    lambda_penalty: float,
    alpha_size: float,
    cost_beta: float,
    random_state: int,
    cv_splits: int = 5,
) -> callable:
    feature_names = list(feature_names)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def fitness(chromosome: np.ndarray, X: np.ndarray, y: ArrayLike) -> float:
        indices = np.where(chromosome == 1)[0]
        if indices.size == 0:
            return -np.inf

        selected_features = [feature_names[idx] for idx in indices]
        subset_penalty = compute_subset_penalty(selected_features, penalty_matrix, ensemble_weights)

        redundancy_term = lambda_penalty * subset_penalty
        size_term = alpha_size * len(selected_features)

        X_subset = X[:, indices]
        y_array = np.asarray(y)

        total_nll = 0.0
        for train_idx, val_idx in skf.split(X_subset, y_array):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            model = LogisticRegression(
                max_iter=1500,
                solver="lbfgs",
                class_weight=None,
                random_state=random_state,
            )

            sample_weight_train = np.where(y_train == 1, cost_beta, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight_train)

            probabilities = model.predict_proba(X_val)[:, 1]
            sample_weight_val = np.where(y_val == 1, cost_beta, 1.0)
            total_nll += cost_sensitive_negative_log_likelihood(y_val, probabilities, sample_weight_val)

        avg_nll = total_nll / cv_splits
        fitness_score = -(avg_nll + redundancy_term + size_term)
        return fitness_score

    return fitness


def optimise_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    beta: float = 2.0,
    sample_weight: ArrayLike | None = None,
) -> Tuple[float, float]:
    def objective(threshold: float) -> float:
        preds = (y_prob >= threshold).astype(int)
        score = fbeta_score(
            y_true,
            preds,
            beta=beta,
            zero_division=0,
            sample_weight=sample_weight,
        )
        return -score

    result = minimize_scalar(
        objective,
        bounds=(0.01, 0.99),
        method="bounded",
        options={"xatol": 1e-3},
    )

    best_threshold = float(result.x)
    best_score = float(-result.fun)
    return best_threshold, best_score


def explain_with_shap(
    model: LogisticRegression,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    max_samples: int = 1000,
) -> None:
    try:
        import shap  # type: ignore
    except ImportError:
        print("[Explainability] SHAP not installed; skipping explanation.")
        return

    sample = X.sample(n=min(max_samples, len(X)), random_state=0)
    explainer = shap.LinearExplainer(model, sample, feature_perturbation="correlation")
    shap_values = explainer.shap_values(sample)
    importances = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    top_features = ranked[:10]
    print("[Explainability] Top SHAP importances:")
    for name, value in top_features:
        print(f"  - {name}: {value:.4f}")


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

        # ------------------------------------------------------------------
        # Stage: Feature Scoring
        # ------------------------------------------------------------------
        ensemble_weights = {
            "pca": args.weight_pca,
            "mutual_info": args.weight_mi,
            "random_forest": args.weight_rf,
        }
        if tracker.is_completed("feature_scoring"):
            tracker.log_event("feature_scoring", "Loading cached feature scores")
            scores = tracker.load_feature_scores()
            pca_scores = scores["pca"]
            mi_scores = scores["mi"]
            rf_scores = scores["rf"]
            ensemble_scores = scores["ensemble"]
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
            tracker.save_feature_scores(pca_scores, mi_scores, rf_scores, ensemble_scores)
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
        if tracker.is_completed("ga"):
            tracker.log_event("ga", "Loading cached GA results")
            ga_data = tracker.load_ga_results()
            selected_features = ga_data["selected_features"]
            ga_score = ga_data.get("best_score")
            redundancy_penalty = ga_data.get("redundancy_penalty")
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
            tracker.save_ga_results(
                selected_features,
                ga_score,
                history=selector.history_,
                redundancy_penalty=redundancy_penalty,
            )
            tracker.log_event(
                "ga",
                "Genetic algorithm completed",
                {
                    "selected_features": len(selected_features),
                    "best_score": ga_score,
                    "redundancy_penalty": redundancy_penalty,
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
        threshold: float
        val_score: float

        if tracker.is_completed("solver"):
            tracker.log_event("solver", "Loading cached solver outputs")
            solver_results = tracker.load_solver_results()
            threshold = solver_results["threshold"]
            val_score = solver_results["val_score"]
            solver_backend = solver_results["backend"]
            solver_details = solver_results["solver_details"]

            if solver_backend == "custom":
                weights = solver_results["weights"]
                bias = solver_results["bias"]
                model = configure_sklearn_like_model(weights, bias, selected_features)
            else:
                model = solver_results["model"]
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
                )

                solver_output = solve_cost_sensitive_logistic(
                    data["X_train_res"][selected_features],
                    data["y_train_res"],
                    train_weights,
                    solver_config,
                )

                weights = solver_output["weights"]
                bias = solver_output["bias"]
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
                }
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
                    weights = solver_results["weights"]
                    bias = solver_results["bias"]
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

        if not args.skip_explainability:
            explain_with_shap(model, data["X_test_scaled"][selected_features], selected_features)

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
    parser = argparse.ArgumentParser(description="Information-theoretic ensemble fraud detection pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the credit card fraud dataset (CSV).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional identifier used for result/log directory names.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a previous run directory or state.json file to resume from.",
    )
    parser.add_argument(
        "--force-new-run",
        action="store_true",
        help="Ignore existing states and always start a new run.",
    )
    parser.add_argument("--target-column", type=str, default="Class", help="Name of the target column in the dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion reserved for the test split.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation split fraction from the training pool.")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for F-beta and cost-sensitive metrics.")
    parser.add_argument("--cost-beta", type=float, default=5.0, help="Sample weight multiplier for the fraud class.")
    parser.add_argument("--weight-pca", type=float, default=0.35, help="Ensemble weight for PCA-based importance.")
    parser.add_argument("--weight-mi", type=float, default=0.35, help="Ensemble weight for mutual information scores.")
    parser.add_argument("--weight-rf", type=float, default=0.30, help="Ensemble weight for random forest importances.")
    parser.add_argument("--penalty-weight-cmi", type=float, default=0.6, help="Weight assigned to conditional mutual information in the redundancy penalty.")
    parser.add_argument("--penalty-weight-corr", type=float, default=0.25, help="Weight assigned to Pearson correlation in the redundancy penalty.")
    parser.add_argument("--penalty-weight-vif", type=float, default=0.15, help="Weight assigned to VIF contributions in the redundancy penalty.")
    parser.add_argument("--redundancy-budget", type=float, default=0.75, help="Maximum cumulative redundancy penalty before rejecting a feature during selection.")
    parser.add_argument("--lambda-penalty", type=float, default=0.05, help="Regularisation strength on redundancy penalty in GA fitness and evaluation.")
    parser.add_argument("--alpha-size", type=float, default=0.01, help="Penalty applied per selected feature in GA fitness and evaluation.")
    parser.add_argument("--min-candidate-features", type=int, default=12, help="Minimum number of features retained after redundancy screening.")
    parser.add_argument("--ga-population", type=int, default=60, help="Population size for the genetic algorithm.")
    parser.add_argument("--ga-generations", type=int, default=40, help="Number of GA generations.")
    parser.add_argument("--ga-mutation", type=float, default=0.08, help="Mutation probability for each gene in GA.")
    parser.add_argument("--ga-min-features", type=int, default=6, help="Minimum number of active features enforced in GA individuals.")
    parser.add_argument("--ga-cv-splits", type=int, default=5, help="Number of CV folds inside GA fitness evaluation.")
    parser.add_argument("--ga-quiet", action="store_true", help="Suppress per-generation GA logs.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    parser.add_argument("--enable-undersampling", action="store_true", help="Enable random undersampling before SMOTE.")
    parser.add_argument("--undersample-ratio", type=float, default=None, help="Majority/minority ratio for undersampling (if enabled).")
    parser.add_argument("--rho-auc", type=float, default=0.35, help="Weight for ROC-AUC in the overall evaluation score.")
    parser.add_argument("--rho-f1", type=float, default=0.25, help="Weight for cost-sensitive F1 in the overall evaluation score.")
    parser.add_argument("--rho-pr", type=float, default=0.2, help="Weight for PR-AUC in the overall evaluation score.")
    parser.add_argument("--rho-gmean", type=float, default=0.1, help="Weight for G-mean in the overall evaluation score.")
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="custom",
        choices=["custom", "sklearn"],
        help="Backend used for optimising logistic parameters (default: custom solver).",
    )
    parser.add_argument("--solver-max-iter", type=int, default=400, help="Maximum iterations for the custom solver.")
    parser.add_argument("--solver-lr", type=float, default=0.1, help="Learning rate for the custom solver.")
    parser.add_argument("--solver-tol", type=float, default=1e-5, help="Tolerance for solver convergence.")
    parser.add_argument("--solver-momentum", type=float, default=0.0, help="Momentum term for solver gradient descent.")
    parser.add_argument(
        "--solver-method",
        type=str,
        default="gd",
        choices=["gd", "momentum", "nesterov", "adam"],
        help="First-order optimisation variant to use.",
    )
    parser.add_argument("--solver-line-search", action="store_true", help="Enable backtracking line search.")
    parser.add_argument("--solver-line-alpha", type=float, default=0.3, help="Armijo alpha parameter for line search.")
    parser.add_argument("--solver-line-beta", type=float, default=0.8, help="Backtracking decay factor for line search.")
    parser.add_argument("--solver-adam-beta1", type=float, default=0.9, help="Adam beta1 parameter.")
    parser.add_argument("--solver-adam-beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--solver-adam-epsilon", type=float, default=1e-8, help="Adam epsilon parameter.")
    parser.add_argument(
        "--solver-second-order",
        type=str,
        default="none",
        choices=["none", "bfgs"],
        help="Enable quasi-Newton optimisation (BFGS).",
    )
    parser.add_argument("--solver-verbose", action="store_true", help="Print solver progress information.")
    parser.add_argument("--solver-track-history", action="store_true", help="Record loss history during solver optimisation.")
    parser.add_argument("--skip-explainability", action="store_true", help="Skip SHAP-based explainability step.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used across the pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()

