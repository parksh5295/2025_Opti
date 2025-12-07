"""Comparison utility functions for solver results."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compare_weights(
    weights1: Optional[np.ndarray],
    weights2: Optional[np.ndarray],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compare two weight vectors."""
    if weights1 is None or weights2 is None:
        return {"error": "One or both weights are None"}
    
    w1 = np.asarray(weights1, dtype=float)
    w2 = np.asarray(weights2, dtype=float)
    
    # Pad to same length if needed
    if len(w1) != len(w2):
        max_len = max(len(w1), len(w2))
        w1_padded = np.pad(w1, (0, max_len - len(w1)), 'constant')
        w2_padded = np.pad(w2, (0, max_len - len(w2)), 'constant')
    else:
        w1_padded = w1
        w2_padded = w2
    
    # Compute differences
    diff = w1_padded - w2_padded
    abs_diff = np.abs(diff)
    
    # Top differences
    top_indices = np.argsort(abs_diff)[::-1][:10]
    top_diffs = []
    if feature_names and len(feature_names) >= len(w1_padded):
        for idx in top_indices:
            if idx < len(feature_names):
                top_diffs.append({
                    "feature": feature_names[idx],
                    "weight1": float(w1_padded[idx]),
                    "weight2": float(w2_padded[idx]),
                    "difference": float(diff[idx]),
                    "abs_difference": float(abs_diff[idx]),
                })
    else:
        for idx in top_indices:
            top_diffs.append({
                "feature_index": int(idx),
                "weight1": float(w1_padded[idx]),
                "weight2": float(w2_padded[idx]),
                "difference": float(diff[idx]),
                "abs_difference": float(abs_diff[idx]),
            })
    
    return {
        "l2_distance": float(np.linalg.norm(diff)),
        "l1_distance": float(np.sum(abs_diff)),
        "cosine_similarity": float(np.dot(w1_padded, w2_padded) / (np.linalg.norm(w1_padded) * np.linalg.norm(w2_padded) + 1e-10)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "top_differences": top_diffs,
    }


def compare_metrics(
    results1: Dict[str, object],
    results2: Dict[str, object],
) -> pd.DataFrame:
    """Compare evaluation metrics between two runs."""
    metrics = []
    
    metric_names = ["roc_auc", "pr_auc", "f1", "overall_score"]
    for metric in metric_names:
        val1 = results1.get(metric, 0.0)
        val2 = results2.get(metric, 0.0)
        diff = val2 - val1
        pct_diff = (diff / val1 * 100) if val1 != 0 else 0.0
        metrics.append({
            "metric": metric,
            f"{results1['method_label']}": val1,
            f"{results2['method_label']}": val2,
            "difference": diff,
            "percent_change": pct_diff,
        })
    
    # Cost-sensitive metrics
    cs1 = results1.get("cost_sensitive", {})
    cs2 = results2.get("cost_sensitive", {})
    for metric in ["precision_cs", "recall_cs", "f1_cs", "g_mean"]:
        val1 = cs1.get(metric, 0.0)
        val2 = cs2.get(metric, 0.0)
        diff = val2 - val1
        pct_diff = (diff / val1 * 100) if val1 != 0 else 0.0
        metrics.append({
            "metric": metric,
            f"{results1['method_label']}": val1,
            f"{results2['method_label']}": val2,
            "difference": diff,
            "percent_change": pct_diff,
        })
    
    return pd.DataFrame(metrics)


def compare_convergence(
    loss_history1: Optional[List[float]],
    loss_history2: Optional[List[float]],
    final_loss1: Optional[float],
    final_loss2: Optional[float],
    iterations1: Optional[int],
    iterations2: Optional[int],
) -> Dict[str, object]:
    """Compare convergence characteristics between two runs.
    
    Based on Chapters 4 (Local Descent), 5 (First-Order Methods), and 6 (Second-Order Methods).
    """
    comparison = {}
    
    # Final loss comparison
    if final_loss1 is not None and final_loss2 is not None:
        comparison["final_loss_diff"] = final_loss2 - final_loss1
        comparison["final_loss_ratio"] = final_loss2 / final_loss1 if final_loss1 != 0 else None
        comparison["final_loss1"] = final_loss1
        comparison["final_loss2"] = final_loss2
    
    # Iteration count comparison
    if iterations1 is not None and iterations2 is not None:
        comparison["iterations_diff"] = iterations2 - iterations1
        comparison["iterations_ratio"] = iterations2 / iterations1 if iterations1 > 0 else None
        comparison["iterations1"] = iterations1
        comparison["iterations2"] = iterations2
    
    # Loss history analysis (if available)
    if loss_history1 and loss_history2:
        # Convergence rate: average loss reduction per iteration
        if len(loss_history1) > 1:
            initial_loss1 = loss_history1[0]
            final_loss1_hist = loss_history1[-1]
            convergence_rate1 = (initial_loss1 - final_loss1_hist) / len(loss_history1)
            comparison["convergence_rate1"] = convergence_rate1
        else:
            comparison["convergence_rate1"] = None
        
        if len(loss_history2) > 1:
            initial_loss2 = loss_history2[0]
            final_loss2_hist = loss_history2[-1]
            convergence_rate2 = (initial_loss2 - final_loss2_hist) / len(loss_history2)
            comparison["convergence_rate2"] = convergence_rate2
        else:
            comparison["convergence_rate2"] = None
        
        # Loss reduction percentage
        if len(loss_history1) > 1 and loss_history1[0] > 0:
            loss_reduction1 = (loss_history1[0] - loss_history1[-1]) / loss_history1[0] * 100
            comparison["loss_reduction1_pct"] = loss_reduction1
        else:
            comparison["loss_reduction1_pct"] = None
        
        if len(loss_history2) > 1 and loss_history2[0] > 0:
            loss_reduction2 = (loss_history2[0] - loss_history2[-1]) / loss_history2[0] * 100
            comparison["loss_reduction2_pct"] = loss_reduction2
        else:
            comparison["loss_reduction2_pct"] = None
        
        # Minimum loss achieved
        comparison["min_loss1"] = min(loss_history1) if loss_history1 else None
        comparison["min_loss2"] = min(loss_history2) if loss_history2 else None
    
    return comparison


def compare_optimization_methods(
    method1: str,
    method2: str,
    learning_rate1: Optional[float],
    learning_rate2: Optional[float],
    line_search1: bool,
    line_search2: bool,
) -> Dict[str, object]:
    """Compare optimization method characteristics.
    
    Based on Chapter 5 (First-Order Methods) and Chapter 4 (Line Search).
    """
    return {
        "method1": method1,
        "method2": method2,
        "methods_same": method1 == method2,
        "learning_rate1": learning_rate1,
        "learning_rate2": learning_rate2,
        "learning_rate_diff": learning_rate2 - learning_rate1 if (learning_rate1 is not None and learning_rate2 is not None) else None,
        "line_search1": line_search1,
        "line_search2": line_search2,
        "line_search_both": line_search1 and line_search2,
    }


def compare_features(
    features1: List[str],
    features2: List[str],
) -> Dict[str, object]:
    """Compare selected features between two runs."""
    set1 = set(features1)
    set2 = set(features2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return {
        "common_features": sorted(list(common)),
        "only_in_first": sorted(list(only1)),
        "only_in_second": sorted(list(only2)),
        "jaccard_similarity": len(common) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0,
        "count_first": len(set1),
        "count_second": len(set2),
        "count_common": len(common),
    }

