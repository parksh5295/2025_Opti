"""Evaluation utilities for the fraud detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class EvaluationConfig:
    """Configuration for cost-sensitive evaluation."""

    beta: float = 2.0
    rho_auc: float = 0.35
    rho_f1: float = 0.25
    rho_pr: float = 0.2
    rho_gmean: float = 0.1
    lambda_penalty: float = 0.05
    alpha_size: float = 0.01

    def validate(self) -> None:
        if self.rho_auc + self.rho_f1 + self.rho_pr + self.rho_gmean > 1.0:
            raise ValueError("Sum of rho weights must not exceed 1.0.")


def _cost_sensitive_counts(matrix: np.ndarray, beta: float) -> Dict[str, float]:
    tn, fp, fn, tp = matrix.ravel()
    precision_cs = tp / (tp + beta * fp) if (tp + beta * fp) > 0 else 0.0
    recall_cs = tp / (tp + beta * fn) if (tp + beta * fn) > 0 else 0.0
    if precision_cs + recall_cs > 0:
        f1_cs = 2 * (precision_cs * recall_cs) / (precision_cs + recall_cs)
    else:
        f1_cs = 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall_cs * specificity)

    return {
        "precision_cs": precision_cs,
        "recall_cs": recall_cs,
        "f1_cs": f1_cs,
        "g_mean": g_mean,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_model(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    evaluation_config: EvaluationConfig,
    redundancy_penalty: float,
    subset_size: int,
) -> Dict[str, object]:
    """Compute evaluation metrics and overall score."""

    evaluation_config.validate()

    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    f1 = f1_score(y_true, predictions, zero_division=0)

    matrix = confusion_matrix(y_true, predictions)
    cs_counts = _cost_sensitive_counts(matrix, evaluation_config.beta)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)

    rho_sum = evaluation_config.rho_auc + evaluation_config.rho_f1 + evaluation_config.rho_pr + evaluation_config.rho_gmean
    residual_weight = 1.0 - rho_sum
    overall_score = (
        evaluation_config.rho_auc * roc_auc
        + evaluation_config.rho_f1 * cs_counts["f1_cs"]
        + evaluation_config.rho_pr * pr_auc
        + evaluation_config.rho_gmean * cs_counts["g_mean"]
        + residual_weight * f1
        - evaluation_config.lambda_penalty * redundancy_penalty
        - evaluation_config.alpha_size * subset_size
    )

    report = classification_report(y_true, predictions, digits=4, zero_division=0)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "cost_sensitive": cs_counts,
        "overall_score": overall_score,
    }


def format_cost_sensitive_summary(metrics: Dict[str, float]) -> str:
    return (
        f"Precision_cs={metrics['precision_cs']:.4f}, "
        f"Recall_cs={metrics['recall_cs']:.4f}, "
        f"F1_cs={metrics['f1_cs']:.4f}, "
        f"G-mean={metrics['g_mean']:.4f}"
    )


__all__ = ["EvaluationConfig", "evaluate_model", "format_cost_sensitive_summary"]

