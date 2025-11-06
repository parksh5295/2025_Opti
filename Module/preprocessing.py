"""Preprocessing utilities for the credit-card fraud pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingConfig:
    """Configuration for dataset preparation.

    Attributes
    ----------
    target_column:
        Name of the binary label column (1 = fraud).
    test_size:
        Fraction reserved for the test set.
    validation_size:
        Fraction of the remaining training data to hold out for validation.
    random_state:
        Random seed used for reproducibility.
    use_smote:
        Apply SMOTE oversampling on the training split when ``True``.
    use_undersampling:
        Randomly undersample the majority class before oversampling when
        ``True``.
    undersample_ratio:
        Desired majority/minority ratio after undersampling. ``None`` keeps the
        default behaviour of :class:`RandomUnderSampler`.
    scaler_cls:
        Callable returning a feature scaler (defaults to
        :class:`sklearn.preprocessing.StandardScaler`).
    scaler_kwargs:
        Optional keyword arguments forwarded to the scaler initialiser.
    beta_cost:
        Cost-sensitive weight applied to the fraud class when computing sample
        weights.
    """

    target_column: str = "Class"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    use_undersampling: bool = False
    undersample_ratio: Optional[float] = None
    scaler_cls: type = StandardScaler
    scaler_kwargs: Optional[Dict[str, Any]] = None
    beta_cost: float = 5.0


def compute_sample_weights(y: pd.Series, beta: float) -> np.ndarray:
    """Return per-sample weights for cost-sensitive learning."""

    weights = np.ones(len(y), dtype=float)
    weights[y == 1] = beta
    return weights


def load_and_preprocess(data_path: Path, config: PreprocessingConfig) -> Dict[str, object]:
    """Load the dataset and produce train/validation/test splits.

    Returns a mapping containing scaled feature matrices, the fitted scaler, and
    any resampled training data required for downstream modelling.
    """

    df = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)

    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in the dataset.")

    features = [col for col in df.columns if col != config.target_column]
    X = df[features]
    y = df[config.target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config.validation_size,
        stratify=y_temp,
        random_state=config.random_state,
    )

    scaler_kwargs = config.scaler_kwargs or {}
    scaler = config.scaler_cls(**scaler_kwargs)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    y_train_series = y_train.reset_index(drop=True)

    if config.use_undersampling:
        rus = RandomUnderSampler(
            sampling_strategy=config.undersample_ratio,
            random_state=config.random_state,
        )
        X_us, y_us = rus.fit_resample(X_train_scaled_df, y_train_series)
    else:
        X_us, y_us = X_train_scaled_df, y_train_series

    if config.use_smote:
        smote = SMOTE(random_state=config.random_state)
        X_res, y_res = smote.fit_resample(X_us, y_us)
    else:
        X_res, y_res = X_us, y_us

    return {
        "feature_columns": features,
        "scaler": scaler,
        "X_train_res": X_res.reset_index(drop=True),
        "y_train_res": y_res.reset_index(drop=True),
        "X_train_scaled": X_train_scaled_df.reset_index(drop=True),
        "y_train": y_train_series.reset_index(drop=True),
        "X_val_scaled": pd.DataFrame(X_val_scaled, columns=features).reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "X_test_scaled": pd.DataFrame(X_test_scaled, columns=features).reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


__all__ = ["PreprocessingConfig", "load_and_preprocess", "compute_sample_weights"]

