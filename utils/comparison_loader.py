"""Result loading and auto-execution utilities for solver comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Module.experiment_tracker import ExperimentTracker


def load_run_results(
    data_path: Path,
    method_label: str,
    run_name: Optional[str] = None,
    auto_run: bool = False,
    extra_args: Optional[List[str]] = None,
    csv_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Load all results from a specific run.
    
    If run_name is None, automatically finds the latest completed run for the method_label.
    If no completed run exists and auto_run=True, automatically runs the pipeline.
    
    Parameters
    ----------
    data_path: Path to data directory or CSV file (for ExperimentTracker)
    method_label: Method label (e.g., 'without_hessian', 'commercial_gurobi')
    run_name: Optional run name. If None, finds latest completed run.
    auto_run: Whether to automatically run pipeline if no completed run found
    extra_args: Optional additional arguments for auto-run
    csv_path: Optional CSV file path for auto-run. If None, will be constructed from data_path.
    """
    # Determine CSV path if not provided and data_path is not a CSV file
    if csv_path is None and data_path.suffix != ".csv":
        # data_path is a directory, construct CSV path
        csv_path = data_path / "creditcard" / "creditcard.csv"
        if not csv_path.exists():
            csv_path = data_path / "creditcard.csv"
    elif csv_path is None and data_path.suffix == ".csv":
        # data_path is already a CSV file
        csv_path = data_path
    
    # If run_name is not specified, find the latest completed run
    actual_method_label = method_label  # Track the actual method_label used
    if run_name is None:
        latest_state = ExperimentTracker.find_latest_completed_state(data_path, method_label)
        if latest_state is None:
            if auto_run:
                print(f"[Comparison] No completed runs found for {method_label}. Running pipeline automatically...")
                run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
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
    # Note: data_path might be a CSV file path (from compare_solvers.py) or a directory
    # ExperimentTracker expects CSV file path (like main.py and main_commercial.py)
    tracker_data_path = data_path
    if csv_path is not None and csv_path.suffix == ".csv":
        # If csv_path is provided and is a CSV file, use it for tracker
        tracker_data_path = csv_path
    elif data_path.suffix != ".csv":
        # If data_path is not a CSV file, construct CSV path for tracker
        tracker_data_path = data_path / "creditcard" / "creditcard.csv"
        if not tracker_data_path.exists():
            tracker_data_path = data_path / "creditcard.csv"
    
    tracker = ExperimentTracker(
        data_path=tracker_data_path,
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
            run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
            # Update tracker with new run
            tracker = ExperimentTracker(
                data_path=data_path,
                method_label=actual_method_label,
                run_name=run_name,
            )
            # Wait for file system sync
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
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
                    tracker = ExperimentTracker(
                        data_path=data_path,
                        method_label=actual_method_label,
                        run_name=run_name,
                    )
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
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
                    # Update tracker with new run
                    tracker = ExperimentTracker(
                        data_path=data_path,
                        method_label=actual_method_label,
                        run_name=run_name,
                    )
                    # Wait a moment for file system to sync
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
            run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
            # Update tracker with new run
            tracker = ExperimentTracker(
                data_path=data_path,
                method_label=actual_method_label,
                run_name=run_name,
            )
            # Wait for file system sync
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
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
                    tracker = ExperimentTracker(
                        data_path=data_path,
                        method_label=actual_method_label,
                        run_name=run_name,
                    )
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
                    run_name, actual_method_label = _auto_run_pipeline(data_path, method_label, extra_args=extra_args, csv_path=csv_path)
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


def _auto_run_pipeline(data_path: Path, method_label: str, extra_args: Optional[List[str]] = None, csv_path: Optional[Path] = None) -> tuple[str, str]:
    """Automatically run the pipeline for a given method_label.
    
    Parameters
    ----------
    data_path: Path to data directory (for ExperimentTracker)
    method_label: Method label (e.g., 'without_hessian', 'commercial_gurobi')
    extra_args: Optional list of additional command-line arguments to pass to the pipeline
    csv_path: Optional CSV file path. If None, will be constructed from data_path.
    
    Returns
    -------
    Tuple of (run_name, actual_method_label) where actual_method_label
    is the method_label that was actually used to save the run.
    """
    # Determine CSV path if not provided
    # Note: data_path might be a CSV file path (from compare_solvers.py) or a directory
    if csv_path is None:
        # If data_path is already a CSV file, use it directly
        if data_path.suffix == ".csv":
            csv_path = data_path
        else:
            # Otherwise, construct from directory
            csv_path = data_path / "creditcard" / "creditcard.csv"
            if not csv_path.exists():
                csv_path = data_path / "creditcard.csv"
    
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
            "--data-path", str(csv_path),
            "--solver", actual_solver,
        ]
    else:
        # Custom solver
        script = "main.py"
        cmd = [
            sys.executable,
            script,
            "--data-path", str(csv_path),
            "--solver-backend", "custom",
        ]
        
        # Determine solver method based on method_label
        if method_label == "with_hessian":
            cmd.extend(["--solver-second-order", "bfgs"])
        elif method_label == "without_hessian":
            cmd.extend(["--solver-method", "adam"])  # Default first-order method
        
        # Add extra arguments if provided
        if extra_args:
            cmd.extend(extra_args)
    
    print(f"[Comparison] Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        print(f"[Comparison] Pipeline execution completed successfully")
        
        # Wait a moment for file system to sync and state to be written
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

