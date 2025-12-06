"""Feature selection utility functions.

This module contains functions for computing feature importance scores,
building redundancy penalty matrices, and performing redundancy-aware selection.
These functions are used in the main pipeline but are separated for better
code organization.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalise feature scores to [0, 1] range."""
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
    """Compute PCA-based feature importance scores."""
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
    """Compute mutual information-based feature importance scores."""
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
    """Compute random forest-based feature importance scores."""
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
    """Combine multiple feature scoring methods using weighted ensemble."""
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
    """Compute conditional mutual information matrix between features."""
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
    """Compute Variance Inflation Factor (VIF) scores for features."""
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
    """Normalise matrix values to [0, 1] range."""
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
    """Build redundancy penalty matrix using CMI, correlation, and VIF."""
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
    """Select features while minimizing redundancy using penalty matrix."""
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
    features: List[str],
    penalty_matrix: pd.DataFrame,
    ensemble_weights: Dict[str, float],
) -> float:
    """Compute redundancy penalty for a subset of features."""
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

