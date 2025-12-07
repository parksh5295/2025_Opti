"""Report generation utilities for solver comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from utils.comparison_utils import (
    compare_convergence,
    compare_features,
    compare_metrics,
    compare_optimization_methods,
    compare_weights,
)


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
    
    # Convergence analysis (Chapters 4, 5, 6)
    loss_history1 = results1.get("loss_history")
    loss_history2 = results2.get("loss_history")
    final_loss1 = results1.get("final_loss")
    final_loss2 = results2.get("final_loss")
    iterations1 = results1.get("iterations")
    iterations2 = results2.get("iterations")
    
    if (loss_history1 or loss_history2 or final_loss1 or final_loss2 or iterations1 or iterations2):
        lines.append("-" * 80)
        lines.append("CONVERGENCE ANALYSIS (Ch. 4, 5, 6)")
        lines.append("-" * 80)
        
        conv_comp = compare_convergence(
            loss_history1, loss_history2,
            final_loss1, final_loss2,
            iterations1, iterations2,
        )
        
        if final_loss1 is not None and final_loss2 is not None:
            lines.append(f"Final Loss: {final_loss1:.6f} vs {final_loss2:.6f} (diff: {conv_comp.get('final_loss_diff', 0):.6f})")
            if conv_comp.get("final_loss_ratio") is not None:
                lines.append(f"  Ratio: {conv_comp['final_loss_ratio']:.4f}x")
        
        if iterations1 is not None and iterations2 is not None:
            lines.append(f"Iterations: {iterations1} vs {iterations2} (diff: {conv_comp.get('iterations_diff', 0)})")
            if conv_comp.get("iterations_ratio") is not None:
                lines.append(f"  Ratio: {conv_comp['iterations_ratio']:.4f}x")
        
        if conv_comp.get("convergence_rate1") is not None:
            lines.append(f"Convergence Rate (Run 1): {conv_comp['convergence_rate1']:.6f} loss/iteration")
        if conv_comp.get("convergence_rate2") is not None:
            lines.append(f"Convergence Rate (Run 2): {conv_comp['convergence_rate2']:.6f} loss/iteration")
        
        if conv_comp.get("loss_reduction1_pct") is not None:
            lines.append(f"Loss Reduction (Run 1): {conv_comp['loss_reduction1_pct']:.2f}%")
        if conv_comp.get("loss_reduction2_pct") is not None:
            lines.append(f"Loss Reduction (Run 2): {conv_comp['loss_reduction2_pct']:.2f}%")
        
        if conv_comp.get("min_loss1") is not None and conv_comp.get("min_loss2") is not None:
            lines.append(f"Minimum Loss Achieved: {conv_comp['min_loss1']:.6f} vs {conv_comp['min_loss2']:.6f}")
        
        lines.append("")
    
    # Optimization method comparison (Chapter 5)
    method1 = results1.get("method", "unknown")
    method2 = results2.get("method", "unknown")
    lr1 = results1.get("learning_rate")
    lr2 = results2.get("learning_rate")
    ls1 = results1.get("line_search", False)
    ls2 = results2.get("line_search", False)
    
    if method1 != "unknown" or method2 != "unknown" or lr1 or lr2:
        lines.append("-" * 80)
        lines.append("OPTIMIZATION METHOD COMPARISON (Ch. 5)")
        lines.append("-" * 80)
        
        method_comp = compare_optimization_methods(method1, method2, lr1, lr2, ls1, ls2)
        
        lines.append(f"Method: {method_comp['method1']} vs {method_comp['method2']}")
        if method_comp["methods_same"]:
            lines.append("  (Same method)")
        else:
            lines.append("  (Different methods)")
        
        if lr1 is not None or lr2 is not None:
            lr1_str = f"{lr1:.6f}" if lr1 is not None else "N/A"
            lr2_str = f"{lr2:.6f}" if lr2 is not None else "N/A"
            lines.append(f"Learning Rate: {lr1_str} vs {lr2_str}")
            if method_comp.get("learning_rate_diff") is not None:
                lines.append(f"  Difference: {method_comp['learning_rate_diff']:.6f}")
        
        lines.append(f"Line Search: {method_comp['line_search1']} vs {method_comp['line_search2']}")
        if method_comp["line_search_both"]:
            lines.append("  (Both use line search - Ch. 4.3)")
        elif method_comp["line_search1"] or method_comp["line_search2"]:
            lines.append("  (One uses line search)")
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

