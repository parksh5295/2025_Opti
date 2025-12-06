"""Compare results from custom solver and commercial solvers.

This script compares performance metrics, weights, selected features, and other
aspects between different solver runs to help understand the differences between
custom and commercial solvers.

Recommended comparison points:
1. Performance Metrics: ROC-AUC, PR-AUC, F1-score, Cost-sensitive metrics
2. Weights Comparison: L2/L1 distance, cosine similarity, top differences
3. Selected Features: Common features, Jaccard similarity
4. Convergence Analysis: Loss history, convergence rate, final loss, iteration count
5. Optimization Method: Method type, learning rate, line search usage
6. Confusion Matrix: TP, TN, FP, FN comparison
7. Optimization Trajectory: Weight space changes, snapshots analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from Module.experiment_tracker import ExperimentTracker
from utils.convergence_plots import (
    plot_combined_convergence,
    plot_ga_convergence,
    plot_solver_convergence,
)


def load_run_results(
    data_path: Path,
    method_label: str,
    run_name: Optional[str] = None,
    auto_run: bool = False,
) -> Dict[str, object]:
    """Load all results from a specific run.
    
    If run_name is None, automatically finds the latest completed run for the method_label.
    If no completed run exists and auto_run=True, automatically runs the pipeline.
    """
    # If run_name is not specified, find the latest completed run
    actual_method_label = method_label  # Track the actual method_label used
    if run_name is None:
        latest_state = ExperimentTracker.find_latest_completed_state(data_path, method_label)
        if latest_state is None:
            if auto_run:
                print(f"[Comparison] No completed runs found for {method_label}. Running pipeline automatically...")
                run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
            else:
                raise FileNotFoundError(
                    f"No completed runs found for method_label: {method_label}. "
                    f"Please specify --run1-name or --run2-name explicitly, or use --auto-run to automatically execute missing runs."
                )
        else:
            # Extract run_name from state file path
            run_name = latest_state.parent.name
            print(f"[Comparison] Auto-selected latest completed run: {run_name} (method: {method_label})")
    
    # Use actual_method_label for tracker (in case alias was used)
    tracker = ExperimentTracker(
        data_path=data_path,
        method_label=actual_method_label,
        run_name=run_name,
    )
    
    results = {
        "method_label": method_label,
        "run_name": run_name,
    }
    
    # Check if solver stage is completed before trying to load
    if not tracker.is_completed("solver"):
        if auto_run:
            print(f"[Comparison] Solver stage not completed for {method_label}/{run_name}. Re-running pipeline...")
            run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
            # Update tracker with new run
            tracker = ExperimentTracker(
                data_path=data_path,
                method_label=actual_method_label,
                run_name=run_name,
            )
            # Wait for file system sync
            import time
            time.sleep(2)
            # Update results with new run info
            results["method_label"] = actual_method_label
            results["run_name"] = run_name
        else:
            results["solver"] = None
            results["error"] = (
                f"Solver stage not completed for {method_label}/{run_name}. "
                f"Use --auto-run to automatically re-run the pipeline."
            )
            # Skip loading solver results - set to empty dict to avoid further processing
            solver_results = {}
    
    # Load solver results
    solver_results = None  # Initialize variable
    if "error" not in results:
        try:
            if not tracker.is_completed("solver"):
                # This should have been caught earlier, but double-check
                if auto_run:
                    print(f"[Comparison] Solver stage still not completed. Re-running pipeline...")
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
                    tracker = ExperimentTracker(
                        data_path=data_path,
                        method_label=actual_method_label,
                        run_name=run_name,
                    )
                    import time
                    time.sleep(2)
                    results["method_label"] = actual_method_label
                    results["run_name"] = run_name
                else:
                    raise KeyError(f"Solver stage not completed for {method_label}/{run_name}")
            
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
                results["final_loss"] = solver_details.get("final_loss", None)
                results["method"] = solver_details.get("method", "unknown")
                results["learning_rate"] = solver_details.get("learning_rate", None)
                results["line_search"] = solver_details.get("line_search", False)
                # Load history if available
                if "history" in solver_details and solver_details["history"]:
                    results["loss_history"] = solver_details["history"].get("loss", [])
                else:
                    results["loss_history"] = None
            elif backend in ["gurobi", "pymoo_ga"]:
                results["iterations"] = None  # Commercial solvers don't report exact iterations
                results["snapshots_count"] = len(solver_results.get("snapshots", []))
                results["final_loss"] = solver_details.get("final_loss", None)
                results["method"] = backend
                results["learning_rate"] = None
                results["line_search"] = False
                results["loss_history"] = None
            else:
                results["iterations"] = None
                results["snapshots_count"] = 0
                results["final_loss"] = None
                results["method"] = backend
                results["learning_rate"] = None
                results["line_search"] = False
                results["loss_history"] = None
        except Exception as e:
        # If loading fails and auto_run is enabled, try to run the pipeline
        if auto_run and run_name is not None:
            print(f"[Comparison] Failed to load solver results for {method_label}/{run_name}: {e}")
            print(f"[Comparison] Error type: {type(e).__name__}")
            import traceback
            print(f"[Comparison] Error details: {traceback.format_exc()}")
            print(f"[Comparison] Attempting to re-run pipeline...")
            try:
                run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
                # Update tracker with new run
                tracker = ExperimentTracker(
                    data_path=data_path,
                    method_label=actual_method_label,
                    run_name=run_name,
                )
                # Wait a moment for file system to sync
                import time
                time.sleep(1)
                # Retry loading
                solver_results = tracker.load_solver_results()
                results["solver"] = solver_results
                backend = solver_results.get("backend", "unknown")
                results["backend"] = backend
                # ... (rest of the loading logic, same as above)
                if backend == "custom":
                    results["weights"] = solver_results.get("weights")
                    results["bias"] = solver_results.get("bias")
                elif backend in ["gurobi", "pymoo_ga", "sklearn"]:
                    results["weights"] = solver_results.get("weights")
                    results["bias"] = solver_results.get("bias")
                    if results["weights"] is None:
                        model = solver_results.get("model")
                        if model is not None:
                            results["weights"] = model.coef_.ravel()
                            results["bias"] = float(model.intercept_[0])
                solver_details = solver_results.get("solver_details", {})
                results["solver_details"] = solver_details
                results["threshold"] = solver_results.get("threshold", 0.5)
                results["val_score"] = solver_results.get("val_score", 0.0)
                results["selected_features"] = solver_details.get("selected_features", [])
                if backend == "custom":
                    results["iterations"] = solver_details.get("iterations", None)
                    results["snapshots_count"] = solver_details.get("snapshots_recorded", 0)
                    results["final_loss"] = solver_details.get("final_loss", None)
                    results["method"] = solver_details.get("method", "unknown")
                    results["learning_rate"] = solver_details.get("learning_rate", None)
                    results["line_search"] = solver_details.get("line_search", False)
                    if "history" in solver_details and solver_details["history"]:
                        results["loss_history"] = solver_details["history"].get("loss", [])
                    else:
                        results["loss_history"] = None
                elif backend in ["gurobi", "pymoo_ga"]:
                    results["iterations"] = None
                    results["snapshots_count"] = len(solver_results.get("snapshots", []))
                    results["final_loss"] = solver_details.get("final_loss", None)
                    results["method"] = backend
                    results["learning_rate"] = None
                    results["line_search"] = False
                    results["loss_history"] = None
                else:
                    results["iterations"] = None
                    results["snapshots_count"] = 0
                    results["final_loss"] = None
                    results["method"] = backend
                    results["learning_rate"] = None
                    results["line_search"] = False
                    results["loss_history"] = None
                results["method_label"] = actual_method_label  # Update method_label
                print(f"[Comparison] Successfully loaded results after re-running pipeline")
            except Exception as retry_error:
                results["solver"] = None
                import traceback
                results["error"] = (
                    f"Initial error: {type(e).__name__}: {e}\n"
                    f"Retry error: {type(retry_error).__name__}: {retry_error}\n"
                    f"Retry traceback: {traceback.format_exc()}"
                )
        else:
            results["solver"] = None
            results["error"] = str(e)
    
    # Check if evaluation stage is completed before trying to load
    if not tracker.is_completed("evaluation"):
        if auto_run and "error" not in results:
            print(f"[Comparison] Evaluation stage not completed for {method_label}/{run_name}. Re-running pipeline...")
            run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
            # Update tracker with new run
            tracker = ExperimentTracker(
                data_path=data_path,
                method_label=actual_method_label,
                run_name=run_name,
            )
            # Wait for file system sync
            import time
            time.sleep(2)
            # Update results with new run info
            results["method_label"] = actual_method_label
            results["run_name"] = run_name
        else:
            results["evaluation"] = None
            results["error_eval"] = (
                f"Evaluation stage not completed for {method_label}/{run_name}. "
                f"Use --auto-run to automatically re-run the pipeline."
            )
            # Skip loading evaluation
            evaluation = None
    
    # Load evaluation results
    evaluation = None  # Initialize variable
    if "error_eval" not in results:
        try:
            if not tracker.is_completed("evaluation"):
                # This should have been caught earlier, but double-check
                if auto_run and "error" not in results:
                    print(f"[Comparison] Evaluation stage still not completed. Re-running pipeline...")
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
                    tracker = ExperimentTracker(
                        data_path=data_path,
                        method_label=actual_method_label,
                        run_name=run_name,
                    )
                    import time
                    time.sleep(2)
                    results["method_label"] = actual_method_label
                    results["run_name"] = run_name
                else:
                    raise KeyError(f"Evaluation stage not completed for {method_label}/{run_name}")
            
            evaluation = tracker.load_evaluation()
            results["evaluation"] = evaluation
            results["roc_auc"] = evaluation.get("roc_auc", 0.0)
            results["pr_auc"] = evaluation.get("pr_auc", 0.0)
            results["f1"] = evaluation.get("f1", 0.0)
            results["overall_score"] = evaluation.get("overall_score", 0.0)
            results["cost_sensitive"] = evaluation.get("cost_sensitive", {})
            results["confusion_matrix"] = evaluation.get("confusion_matrix", None)
        except Exception as e:
        # If loading fails and auto_run is enabled, try to run the pipeline
        if auto_run and run_name is not None and "error" not in results:
            print(f"[Comparison] Failed to load evaluation results for {method_label}/{run_name}: {e}")
            print(f"[Comparison] Attempting to re-run pipeline...")
            try:
                run_name, actual_method_label = _auto_run_pipeline(data_path, method_label)
                # Update tracker with new run
                tracker = ExperimentTracker(
                    data_path=data_path,
                    method_label=actual_method_label,
                    run_name=run_name,
                )
                # Retry loading evaluation
                evaluation = tracker.load_evaluation()
                results["evaluation"] = evaluation
                results["roc_auc"] = evaluation.get("roc_auc", 0.0)
                results["pr_auc"] = evaluation.get("pr_auc", 0.0)
                results["f1"] = evaluation.get("f1", 0.0)
                results["overall_score"] = evaluation.get("overall_score", 0.0)
                results["cost_sensitive"] = evaluation.get("cost_sensitive", {})
                results["confusion_matrix"] = evaluation.get("confusion_matrix", None)
                results["method_label"] = actual_method_label  # Update method_label
                print(f"[Comparison] Successfully loaded evaluation after re-running pipeline")
            except Exception as retry_error:
                results["evaluation"] = None
                import traceback
                results["error_eval"] = (
                    f"Initial error: {type(e).__name__}: {e}\n"
                    f"Retry error: {type(retry_error).__name__}: {retry_error}\n"
                    f"Retry traceback: {traceback.format_exc()}"
                )
        else:
            results["evaluation"] = None
            results["error_eval"] = str(e)
    
    return results


def _auto_run_pipeline(data_path: Path, method_label: str) -> tuple[str, str]:
    """Automatically run the pipeline for a given method_label.
    
    Returns a tuple of (run_name, actual_method_label) where actual_method_label
    is the method_label that was actually used to save the run.
    """
    # Determine which script to run based on method_label
    actual_method_label = method_label  # Track the actual method_label used
    
    if method_label.startswith("commercial_"):
        # Commercial solver
        solver_name = method_label.replace("commercial_", "")
        # Map common aliases to actual solver names
        solver_map = {
            "pymoo": "pymoo_ga",  # Allow pymoo as alias for pymoo_ga
            "gurobi": "gurobi",
            "sklearn": "sklearn",
        }
        actual_solver = solver_map.get(solver_name, solver_name)
        # Update actual_method_label to match what will be saved
        actual_method_label = f"commercial_{actual_solver}"
        
        script = "main_commercial.py"
        cmd = [
            sys.executable,
            script,
            "--data-path", str(data_path),
            "--solver", actual_solver,
        ]
    else:
        # Custom solver
        script = "main.py"
        cmd = [
            sys.executable,
            script,
            "--data-path", str(data_path),
            "--solver-backend", "custom",
        ]
        
        # Determine solver method based on method_label
        if method_label == "with_hessian":
            cmd.extend(["--solver-second-order", "bfgs"])
        elif method_label == "without_hessian":
            cmd.extend(["--solver-method", "adam"])  # Default first-order method
    
    print(f"[Comparison] Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        print(f"[Comparison] Pipeline execution completed successfully")
        
        # Wait a moment for file system to sync and state to be written
        import time
        time.sleep(2)
        
        # Find the newly created run using the actual method_label
        # Try multiple times in case the state file hasn't been written yet
        latest_state = None
        for attempt in range(5):
            latest_state = ExperimentTracker.find_latest_completed_state(data_path, actual_method_label)
            if latest_state is not None:
                break
            if attempt < 4:
                print(f"[Comparison] Waiting for run to complete... (attempt {attempt + 1}/5)")
                time.sleep(2)
        
        if latest_state is None:
            # Try to find any run (even if not completed) to get the run_name
            data_root = ExperimentTracker.compute_data_root(data_path)
            log_root = data_root / "log" / actual_method_label
            if log_root.exists():
                # Find the most recent run directory
                run_dirs = sorted(log_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                if run_dirs:
                    latest_run_dir = run_dirs[0]
                    run_name = latest_run_dir.name
                    state_path = latest_run_dir / "state.json"
                    if state_path.exists():
                        import json
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        status = state.get("status", "unknown")
                        print(f"[Comparison] Found run {run_name} with status: {status}")
                        if status != "completed":
                            raise RuntimeError(
                                f"Pipeline executed but run {run_name} is not completed (status: {status}). "
                                f"Please wait for the pipeline to finish or check the logs."
                            )
                        return run_name, actual_method_label
            
            raise RuntimeError(
                f"Pipeline executed but no completed run found for {actual_method_label}. "
                f"Note: If you used an alias (e.g., 'commercial_pymoo'), the run was saved as '{actual_method_label}'. "
                f"Please check the logs at {log_root if log_root.exists() else 'N/A'}."
            )
        
        run_name = latest_state.parent.name
        print(f"[Comparison] New run created: {run_name} (method: {actual_method_label})")
        return run_name, actual_method_label
        
    except subprocess.CalledProcessError as e:
        print(f"[Comparison] Error executing pipeline: {e}")
        print(f"[Comparison] stdout: {e.stdout}")
        print(f"[Comparison] stderr: {e.stderr}")
        
        # Check for specific error types and provide helpful messages
        error_output = (e.stdout + "\n" + e.stderr).lower()
        
        if "gurobi" in error_output and ("license" in error_output or "size-limited" in error_output):
            raise RuntimeError(
                f"Failed to auto-run pipeline for {method_label}.\n"
                f"Gurobi license limitation: The model is too large for the size-limited license.\n"
                f"Please either:\n"
                f"  1. Use a different solver (e.g., --run2-method commercial_sklearn or commercial_pymoo_ga)\n"
                f"  2. Manually run the pipeline with a smaller dataset or different configuration\n"
                f"  3. Obtain an unrestricted Gurobi license"
            ) from e
        elif "invalid choice" in error_output or "argument --solver" in error_output:
            raise RuntimeError(
                f"Failed to auto-run pipeline for {method_label}.\n"
                f"Invalid solver name. Valid choices are: 'gurobi', 'pymoo_ga', 'sklearn'.\n"
                f"Note: Use 'commercial_pymoo_ga' (not 'commercial_pymoo') for pymoo solver.\n"
                f"Please use a valid method_label (e.g., --run2-method commercial_pymoo_ga or commercial_sklearn)."
            ) from e
        elif "gurobi" in error_output:
            raise RuntimeError(
                f"Failed to auto-run pipeline for {method_label}.\n"
                f"Gurobi error occurred. Please check the error messages above.\n"
                f"Consider using a different solver (e.g., --run2-method commercial_sklearn or commercial_pymoo_ga)."
            ) from e
        else:
            raise RuntimeError(
                f"Failed to auto-run pipeline for {method_label}.\n"
                f"Please check the error messages above and run the pipeline manually:\n"
                f"  {' '.join(cmd)}"
            ) from e


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
    parser.add_argument(
        "--auto-run",
        action="store_true",
        default=True,
        help="Automatically run pipelines for missing runs instead of raising an error (default: True).",
    )
    parser.add_argument(
        "--no-auto-run",
        dest="auto_run",
        action="store_false",
        help="Disable automatic pipeline execution for missing runs.",
    )
    parser.add_argument(
        "--plot-convergence",
        action="store_true",
        help="Generate convergence plots for both runs and save side-by-side comparison.",
    )
    
    args = parser.parse_args()
    
    # Load run 1
    if args.run1_name:
        print(f"[Comparison] Loading Run 1: {args.run1_method} / {args.run1_name}")
    else:
        print(f"[Comparison] Loading Run 1: {args.run1_method} (auto-selecting latest completed run)")
    results1 = load_run_results(args.data_path, args.run1_method, args.run1_name, auto_run=args.auto_run)
    
    # Load run 2
    if args.run2_name:
        print(f"[Comparison] Loading Run 2: {args.run2_method} / {args.run2_name}")
    else:
        print(f"[Comparison] Loading Run 2: {args.run2_method} (auto-selecting latest completed run)")
    results2 = load_run_results(args.data_path, args.run2_method, args.run2_name, auto_run=args.auto_run)
    
    # Check for errors
    if "error" in results1:
        print(f"[Comparison] Error loading Run 1: {results1['error']}")
        return
    if "error" in results2:
        print(f"[Comparison] Error loading Run 2: {results2['error']}")
        return
    
    # Generate comparison report
    report = generate_comparison_report(results1, results2, args.output)
    
    # Generate convergence plots if requested
    if args.plot_convergence:
        try:
            # Load GA history if available
            ga_history1 = None
            ga_history2 = None
            try:
                tracker1 = ExperimentTracker(
                    data_path=args.data_path,
                    method_label=results1["method_label"],
                    run_name=results1["run_name"],
                )
                if tracker1.is_completed("ga"):
                    ga_data1 = tracker1.load_ga_results()
                    ga_history1 = ga_data1.get("history")
            except Exception:
                pass
            
            try:
                tracker2 = ExperimentTracker(
                    data_path=args.data_path,
                    method_label=results2["method_label"],
                    run_name=results2["run_name"],
                )
                if tracker2.is_completed("ga"):
                    ga_data2 = tracker2.load_ga_results()
                    ga_history2 = ga_data2.get("history")
            except Exception:
                pass
            
            # Load solver history if available
            solver_history1 = results1.get("loss_history")
            solver_history2 = results2.get("loss_history")
            
            # Create output directory for plots
            if args.output:
                plot_dir = args.output.parent / f"{args.output.stem}_plots"
            else:
                from Module.experiment_tracker import ExperimentTracker
                data_root = ExperimentTracker.compute_data_root(args.data_path)
                plot_dir = data_root / "comparison_plots" / f"{results1['run_name']}_vs_{results2['run_name']}"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot combined convergence if both GA and Solver histories are available
            if (ga_history1 or solver_history1) and (ga_history2 or solver_history2):
                combined_path = plot_dir / "combined_convergence.png"
                plot_combined_convergence(
                    ga_history1,
                    {"loss": solver_history1} if solver_history1 else None,
                    combined_path,
                    ga_title=f"GA Convergence ({results1['method_label']})",
                    solver_title=f"Solver Convergence ({results1['method_label']})",
                    solver_method=results1.get("method"),
                )
                print(f"[Comparison] Combined convergence plot saved to: {combined_path}")
            
            # Plot individual convergence plots
            if ga_history1:
                ga_plot1 = plot_dir / f"ga_convergence_{results1['method_label']}.png"
                plot_ga_convergence(
                    ga_history1,
                    ga_plot1,
                    title=f"GA Convergence - {results1['method_label']}",
                )
                print(f"[Comparison] GA convergence plot 1 saved to: {ga_plot1}")
            
            if ga_history2:
                ga_plot2 = plot_dir / f"ga_convergence_{results2['method_label']}.png"
                plot_ga_convergence(
                    ga_history2,
                    ga_plot2,
                    title=f"GA Convergence - {results2['method_label']}",
                )
                print(f"[Comparison] GA convergence plot 2 saved to: {ga_plot2}")
            
            if solver_history1:
                solver_plot1 = plot_dir / f"solver_convergence_{results1['method_label']}.png"
                plot_solver_convergence(
                    {"loss": solver_history1},
                    solver_plot1,
                    title=f"Solver Convergence - {results1['method_label']}",
                    method=results1.get("method"),
                )
                print(f"[Comparison] Solver convergence plot 1 saved to: {solver_plot1}")
            
            if solver_history2:
                solver_plot2 = plot_dir / f"solver_convergence_{results2['method_label']}.png"
                plot_solver_convergence(
                    {"loss": solver_history2},
                    solver_plot2,
                    title=f"Solver Convergence - {results2['method_label']}",
                    method=results2.get("method"),
                )
                print(f"[Comparison] Solver convergence plot 2 saved to: {solver_plot2}")
                
        except Exception as e:
            print(f"[Comparison] Warning: Failed to generate convergence plots: {e}")
            import traceback
            traceback.print_exc()
    
    if args.output is None:
        print("\n" + report)
    else:
        print(f"\n[Comparison] Comparison complete. Report saved to: {args.output}")


if __name__ == "__main__":
    main()

