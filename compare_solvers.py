"""Compare results from custom solver and commercial solvers.

This script compares performance metrics, weights, selected features, and other
aspects between different solver runs to help understand the differences between
custom and commercial solvers.

Recommended comparison points:
1. Performance Metrics: ROC-AUC, PR-AUC, F1-score, Cost-sensitive metrics
2. Weights Comparison: L2/L1 distance, cosine similarity, top differences
3. Selected Features: Common features, Jaccard similarity
4. Convergence: Iteration count, snapshots
5. Confusion Matrix: TP, TN, FP, FN comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from Module import ExperimentTracker


def load_run_results(
    data_path: Path,
    method_label: str,
    run_name: Optional[str] = None,
) -> Dict[str, object]:
    """Load all results from a specific run.
    
    If run_name is None, automatically finds the latest completed run for the method_label.
    """
    # If run_name is not specified, find the latest completed run
    if run_name is None:
        latest_state = ExperimentTracker.find_latest_completed_state(data_path, method_label)
        if latest_state is None:
            raise FileNotFoundError(
                f"No completed runs found for method_label: {method_label}. "
                f"Please specify --run1-name or --run2-name explicitly, or ensure there is at least one completed run."
            )
        # Extract run_name from state file path
        run_name = latest_state.parent.name
        print(f"[Comparison] Auto-selected latest completed run: {run_name} (method: {method_label})")
    
    tracker = ExperimentTracker(
        data_path=data_path,
        method_label=method_label,
        run_name=run_name,
    )
    
    results = {
        "method_label": method_label,
        "run_name": run_name,
    }
    
    # Load solver results
    try:
        solver_results = tracker.load_solver_results()
        results["solver"] = solver_results
        backend = solver_results.get("backend", "unknown")
        results["backend"] = backend
        
        # Extract weights and bias
        if backend == "custom":
            results["weights"] = solver_results.get("weights")
            results["bias"] = solver_results.get("bias")
        elif backend in ["gurobi", "pymoo_ga", "sklearn"]:
            # Commercial solvers also save weights/bias
            results["weights"] = solver_results.get("weights")
            results["bias"] = solver_results.get("bias")
            
            # Fallback to model if weights not available (for sklearn)
            if results["weights"] is None:
                model = solver_results.get("model")
                if model is not None:
                    results["weights"] = model.coef_.ravel()
                    results["bias"] = float(model.intercept_[0])
        
        # Solver details
        solver_details = solver_results.get("solver_details", {})
        results["solver_details"] = solver_details
        results["threshold"] = solver_results.get("threshold", 0.5)
        results["val_score"] = solver_results.get("val_score", 0.0)
        
        # Selected features
        results["selected_features"] = solver_details.get("selected_features", [])
        
        # Iteration info
        if backend == "custom":
            results["iterations"] = solver_details.get("iterations", None)
            results["snapshots_count"] = solver_details.get("snapshots_recorded", 0)
        elif backend in ["gurobi", "pymoo_ga"]:
            results["iterations"] = None  # Commercial solvers don't report exact iterations
            results["snapshots_count"] = len(solver_results.get("snapshots", []))
        else:
            results["iterations"] = None
            results["snapshots_count"] = 0
    except Exception as e:
        results["solver"] = None
        results["error"] = str(e)
    
    # Load evaluation results
    try:
        evaluation = tracker.load_evaluation()
        results["evaluation"] = evaluation
        results["roc_auc"] = evaluation.get("roc_auc", 0.0)
        results["pr_auc"] = evaluation.get("pr_auc", 0.0)
        results["f1"] = evaluation.get("f1", 0.0)
        results["overall_score"] = evaluation.get("overall_score", 0.0)
        results["cost_sensitive"] = evaluation.get("cost_sensitive", {})
        results["confusion_matrix"] = evaluation.get("confusion_matrix", None)
    except Exception as e:
        results["evaluation"] = None
        results["error_eval"] = str(e)
    
    return results


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


def generate_comparison_report(
    results1: Dict[str, object],
    results2: Dict[str, object],
    output_path: Optional[Path] = None,
) -> str:
    """Generate a comprehensive comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("SOLVER COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Basic info
    lines.append(f"Run 1: {results1['method_label']} / {results1['run_name']}")
    lines.append(f"  Backend: {results1.get('backend', 'unknown')}")
    lines.append(f"Run 2: {results2['method_label']} / {results2['run_name']}")
    lines.append(f"  Backend: {results2.get('backend', 'unknown')}")
    lines.append("")
    
    # Performance metrics comparison
    lines.append("-" * 80)
    lines.append("PERFORMANCE METRICS COMPARISON")
    lines.append("-" * 80)
    metrics_df = compare_metrics(results1, results2)
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    
    # Cost-sensitive metrics
    cs1 = results1.get("cost_sensitive", {})
    cs2 = results2.get("cost_sensitive", {})
    if cs1 and cs2:
        lines.append("Cost-Sensitive Metrics:")
        lines.append(f"  Precision_cs: {cs1.get('precision_cs', 0.0):.4f} vs {cs2.get('precision_cs', 0.0):.4f}")
        lines.append(f"  Recall_cs: {cs1.get('recall_cs', 0.0):.4f} vs {cs2.get('recall_cs', 0.0):.4f}")
        lines.append(f"  F1_cs: {cs1.get('f1_cs', 0.0):.4f} vs {cs2.get('f1_cs', 0.0):.4f}")
        lines.append(f"  G-mean: {cs1.get('g_mean', 0.0):.4f} vs {cs2.get('g_mean', 0.0):.4f}")
        lines.append("")
    
    # Confusion matrix comparison
    cm1 = results1.get("confusion_matrix")
    cm2 = results2.get("confusion_matrix")
    if cm1 is not None and cm2 is not None:
        lines.append("-" * 80)
        lines.append("CONFUSION MATRIX COMPARISON")
        lines.append("-" * 80)
        lines.append(f"Run 1 ({results1['method_label']}):")
        lines.append(f"  TN={cm1[0,0]}, FP={cm1[0,1]}, FN={cm1[1,0]}, TP={cm1[1,1]}")
        lines.append(f"Run 2 ({results2['method_label']}):")
        lines.append(f"  TN={cm2[0,0]}, FP={cm2[0,1]}, FN={cm2[1,0]}, TP={cm2[1,1]}")
        lines.append("")
    
    # Weights comparison
    weights1 = results1.get("weights")
    weights2 = results2.get("weights")
    if weights1 is not None and weights2 is not None:
        lines.append("-" * 80)
        lines.append("WEIGHTS COMPARISON")
        lines.append("-" * 80)
        
        feature_names = results1.get("selected_features") or results2.get("selected_features")
        weight_comp = compare_weights(weights1, weights2, feature_names)
        
        if "error" not in weight_comp:
            lines.append(f"L2 Distance: {weight_comp['l2_distance']:.6f}")
            lines.append(f"L1 Distance: {weight_comp['l1_distance']:.6f}")
            lines.append(f"Cosine Similarity: {weight_comp['cosine_similarity']:.6f}")
            lines.append(f"Mean Absolute Difference: {weight_comp['mean_abs_diff']:.6f}")
            lines.append(f"Max Absolute Difference: {weight_comp['max_abs_diff']:.6f}")
            lines.append("")
            lines.append("Top 10 Feature Weight Differences:")
            for diff in weight_comp["top_differences"][:10]:
                if "feature" in diff:
                    lines.append(f"  {diff['feature']}: {diff['weight1']:.6f} vs {diff['weight2']:.6f} (diff: {diff['difference']:.6f})")
                else:
                    lines.append(f"  Feature[{diff['feature_index']}]: {diff['weight1']:.6f} vs {diff['weight2']:.6f} (diff: {diff['difference']:.6f})")
            lines.append("")
        
        # Bias comparison
        bias1 = results1.get("bias")
        bias2 = results2.get("bias")
        if bias1 is not None and bias2 is not None:
            lines.append(f"Bias: {bias1:.6f} vs {bias2:.6f} (diff: {bias2 - bias1:.6f})")
            lines.append("")
    
    # Features comparison
    features1 = results1.get("selected_features", [])
    features2 = results2.get("selected_features", [])
    if features1 and features2:
        lines.append("-" * 80)
        lines.append("SELECTED FEATURES COMPARISON")
        lines.append("-" * 80)
        feat_comp = compare_features(features1, features2)
        lines.append(f"Common features: {feat_comp['count_common']} / {feat_comp['count_first']} vs {feat_comp['count_second']}")
        lines.append(f"Jaccard Similarity: {feat_comp['jaccard_similarity']:.4f}")
        lines.append("")
        if feat_comp["only_in_first"]:
            lines.append(f"Only in {results1['method_label']}: {', '.join(feat_comp['only_in_first'][:10])}")
            if len(feat_comp["only_in_first"]) > 10:
                lines.append(f"  ... and {len(feat_comp['only_in_first']) - 10} more")
        if feat_comp["only_in_second"]:
            lines.append(f"Only in {results2['method_label']}: {', '.join(feat_comp['only_in_second'][:10])}")
            if len(feat_comp["only_in_second"]) > 10:
                lines.append(f"  ... and {len(feat_comp['only_in_second']) - 10} more")
        lines.append("")
    
    # Solver details
    lines.append("-" * 80)
    lines.append("SOLVER DETAILS")
    lines.append("-" * 80)
    details1 = results1.get("solver_details", {})
    details2 = results2.get("solver_details", {})
    
    lines.append(f"Run 1 ({results1['method_label']}):")
    lines.append(f"  Threshold: {results1.get('threshold', 'N/A')}")
    lines.append(f"  Val Score: {results1.get('val_score', 'N/A'):.4f}" if results1.get('val_score') is not None else "  Val Score: N/A")
    if results1.get("iterations") is not None:
        lines.append(f"  Iterations: {results1['iterations']}")
    if results1.get("snapshots_count", 0) > 0:
        lines.append(f"  Snapshots: {results1['snapshots_count']}")
    lines.append("")
    
    lines.append(f"Run 2 ({results2['method_label']}):")
    lines.append(f"  Threshold: {results2.get('threshold', 'N/A')}")
    lines.append(f"  Val Score: {results2.get('val_score', 'N/A'):.4f}" if results2.get('val_score') is not None else "  Val Score: N/A")
    if results2.get("iterations") is not None:
        lines.append(f"  Iterations: {results2['iterations']}")
    if results2.get("snapshots_count", 0) > 0:
        lines.append(f"  Snapshots: {results2['snapshots_count']}")
    lines.append("")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        output_path.write_text(report, encoding="utf-8")
        print(f"[Comparison] Report saved to: {output_path}")
    
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare results from custom solver and commercial solvers."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the data directory (parent of Result/ and log/).",
    )
    parser.add_argument(
        "--run1-method",
        type=str,
        required=True,
        help="Method label for first run (e.g., 'without_hessian', 'commercial_gurobi').",
    )
    parser.add_argument(
        "--run1-name",
        type=str,
        default=None,
        help="Run name for first run. If not specified, uses the latest completed run for --run1-method.",
    )
    parser.add_argument(
        "--run2-method",
        type=str,
        required=True,
        help="Method label for second run (e.g., 'with_hessian', 'commercial_pymoo_ga').",
    )
    parser.add_argument(
        "--run2-name",
        type=str,
        default=None,
        help="Run name for second run. If not specified, uses the latest completed run for --run2-method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for comparison report. If not specified, prints to stdout.",
    )
    
    args = parser.parse_args()
    
    # Load run 1
    if args.run1_name:
        print(f"[Comparison] Loading Run 1: {args.run1_method} / {args.run1_name}")
    else:
        print(f"[Comparison] Loading Run 1: {args.run1_method} (auto-selecting latest completed run)")
    results1 = load_run_results(args.data_path, args.run1_method, args.run1_name)
    
    # Load run 2
    if args.run2_name:
        print(f"[Comparison] Loading Run 2: {args.run2_method} / {args.run2_name}")
    else:
        print(f"[Comparison] Loading Run 2: {args.run2_method} (auto-selecting latest completed run)")
    results2 = load_run_results(args.data_path, args.run2_method, args.run2_name)
    
    # Check for errors
    if "error" in results1:
        print(f"[Comparison] Error loading Run 1: {results1['error']}")
        return
    if "error" in results2:
        print(f"[Comparison] Error loading Run 2: {results2['error']}")
        return
    
    # Generate comparison report
    report = generate_comparison_report(results1, results2, args.output)
    
    if args.output is None:
        print("\n" + report)
    else:
        print(f"\n[Comparison] Comparison complete. Report saved to: {args.output}")


if __name__ == "__main__":
    main()

