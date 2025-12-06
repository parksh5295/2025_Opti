"""Explainability utilities for model interpretation.

This module contains functions for generating model explanations using
SHAP and other interpretability methods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def explain_with_shap(
    model: LogisticRegression,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    max_samples: int = 1000,
) -> None:
    """Generate SHAP explanations for model predictions."""
    try:
        import shap  # type: ignore
    except ImportError:
        print("[Explainability] SHAP not installed; skipping explanation.")
        return

    sample = X.sample(n=min(max_samples, len(X)), random_state=0)
    explainer = shap.LinearExplainer(
        model,
        sample,
        feature_perturbation="correlation_dependent",
    )
    shap_values = explainer.shap_values(sample)
    importances = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    top_features = ranked[:10]
    print("[Explainability] Top SHAP importances:")
    for name, value in top_features:
        print(f"  - {name}: {value:.4f}")

