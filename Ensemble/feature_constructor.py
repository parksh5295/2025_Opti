"""Feature construction utilities for ensemble-based transformations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _select_top_features(scores: Dict[str, float], top_k: int) -> List[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def construct_ensemble_features(
    data: Dict[str, pd.DataFrame],
    pca_scores: Dict[str, float],
    mi_scores: Dict[str, float],
    rf_scores: Dict[str, float],
    ensemble_scores: Dict[str, float],
    top_k: int = 5,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    """Create additional features based on top-ranked features per method.

    Parameters
    ----------
    data:
        Mapping containing the aligned datasets (train/validation/test). Each value
        is expected to be a pandas DataFrame with identical columns.
    pca_scores, mi_scores, rf_scores, ensemble_scores:
        Feature importance mappings for each base method and the final ensemble.
    top_k:
        Number of top-ranked features per method to include in the construction.

    Returns
    -------
    constructed_frames:
        Dictionary containing the newly constructed feature columns for each dataset.
    metadata:
        Information describing created columns, top features, and synthetic scores that
        can be merged back into the ensemble scoring workflow.
    """

    datasets = list(data.keys())
    constructed_frames: Dict[str, pd.DataFrame] = {
        key: pd.DataFrame(index=data[key].index) for key in datasets
    }

    method_scores = {
        "pca": pca_scores,
        "mi": mi_scores,
        "rf": rf_scores,
    }

    created_columns: List[str] = []
    top_feature_map: Dict[str, List[str]] = {}
    synthetic_scores: Dict[str, float] = {}

    for method, scores in method_scores.items():
        if not scores:
            continue

        selected = [feat for feat in _select_top_features(scores, top_k) if feat in data[datasets[0]].columns]
        if not selected:
            continue

        top_feature_map[method] = selected

        weight_values = np.array([scores[feat] for feat in selected], dtype=float)
        if weight_values.sum() <= 0:
            weight_values = np.ones_like(weight_values)
        normalised_weights = weight_values / weight_values.sum()

        weighted_col = f"{method.upper()}_WEIGHTED_SUM"
        mean_col = f"{method.upper()}_TOP{len(selected)}_MEAN"
        std_col = f"{method.upper()}_TOP{len(selected)}_STD"

        for key in datasets:
            df = data[key]
            constructed_frames[key][weighted_col] = df[selected].to_numpy() @ normalised_weights
            constructed_frames[key][mean_col] = df[selected].mean(axis=1)
            constructed_frames[key][std_col] = df[selected].std(axis=1, ddof=0)

        created_columns.extend([weighted_col, mean_col, std_col])
        synthetic_scores[weighted_col] = float(np.mean([ensemble_scores.get(feat, 0.0) for feat in selected]))
        synthetic_scores[mean_col] = synthetic_scores[weighted_col]
        synthetic_scores[std_col] = synthetic_scores[weighted_col] / (len(selected) or 1)

    # Construct global ensemble features from combined top features
    combined_top = [feat for feat, _ in sorted(ensemble_scores.items(), key=lambda item: item[1], reverse=True)[:top_k] if feat in data[datasets[0]].columns]

    if combined_top:
        weight_values = np.array([ensemble_scores[feat] for feat in combined_top], dtype=float)
        if weight_values.sum() <= 0:
            weight_values = np.ones_like(weight_values)
        normalised_weights = weight_values / weight_values.sum()

        weighted_col = f"ENSEMBLE_WEIGHTED_SUM"
        mean_col = f"ENSEMBLE_TOP{len(combined_top)}_MEAN"

        for key in datasets:
            df = data[key]
            constructed_frames[key][weighted_col] = df[combined_top].to_numpy() @ normalised_weights
            constructed_frames[key][mean_col] = df[combined_top].mean(axis=1)

        created_columns.extend([weighted_col, mean_col])
        synthetic_scores[weighted_col] = float(np.mean([ensemble_scores[feat] for feat in combined_top]))
        synthetic_scores[mean_col] = synthetic_scores[weighted_col]
        top_feature_map["ensemble"] = combined_top

    metadata = {
        "created_columns": created_columns,
        "top_features": top_feature_map,
        "new_scores": synthetic_scores,
        "top_k": top_k,
    }

    constructed_frames = {key: frame for key, frame in constructed_frames.items() if not frame.empty}

    return constructed_frames, metadata


__all__ = ["construct_ensemble_features"]

