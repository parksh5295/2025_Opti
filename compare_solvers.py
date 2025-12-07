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
from pathlib import Path

from Module.experiment_tracker import ExperimentTracker
from utils.comparison_loader import load_run_results
from utils.comparison_report import generate_comparison_report
from utils.convergence_plots import (
    plot_combined_convergence,
    plot_ga_convergence,
    plot_solver_convergence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare results from custom solver and commercial solvers."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../Data/creditcard/creditcard.csv"),
        help="Path to the CSV file or data directory. Default: ../Data/creditcard/creditcard.csv",
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
    parser.add_argument(
        "--run1-extra-args",
        type=str,
        nargs="*",
        default=None,
        help="Additional command-line arguments to pass when auto-running run1 (e.g., --use-advanced-ga --ga-multi-objective).",
    )
    parser.add_argument(
        "--run2-extra-args",
        type=str,
        nargs="*",
        default=None,
        help="Additional command-line arguments to pass when auto-running run2 (e.g., --use-advanced-ga --ga-multi-objective).",
    )
    
    args = parser.parse_args()
    
    # Normalize data_path: if it's a CSV file, use its parent directory
    # compare_solvers.py uses data directory for ExperimentTracker, but auto-run needs CSV path
    input_path = args.data_path.resolve()
    if input_path.suffix == ".csv":
        # CSV file provided: use parent directory for tracker, but keep CSV path for auto-run
        data_path = input_path.parent
        csv_path_for_auto_run = input_path
        print(f"[Comparison] Detected CSV file path: {input_path}")
        print(f"[Comparison] Using parent directory for tracker: {data_path}")
    else:
        # Directory provided: construct CSV path for auto-run
        data_path = input_path
        csv_path_for_auto_run = data_path / "creditcard" / "creditcard.csv"
        if not csv_path_for_auto_run.exists():
            csv_path_for_auto_run = data_path / "creditcard.csv"
        print(f"[Comparison] Using data directory: {data_path}")
        print(f"[Comparison] CSV path for auto-run: {csv_path_for_auto_run}")
    
    # Resolve relative path
    data_path = data_path.resolve()
    
    # Pass csv_path to load_run_results for auto-run
    # Note: csv_path_for_auto_run is only used when auto-running pipelines
    csv_path_for_auto_run = csv_path_for_auto_run.resolve() if csv_path_for_auto_run.exists() else None
    
    # Load run 1
    if args.run1_name:
        print(f"[Comparison] Loading Run 1: {args.run1_method} / {args.run1_name}")
    else:
        print(f"[Comparison] Loading Run 1: {args.run1_method} (auto-selecting latest completed run)")
    results1 = load_run_results(
        data_path, 
        args.run1_method, 
        args.run1_name, 
        auto_run=args.auto_run,
        extra_args=args.run1_extra_args,
        csv_path=csv_path_for_auto_run,
    )
    
    # Load run 2
    if args.run2_name:
        print(f"[Comparison] Loading Run 2: {args.run2_method} / {args.run2_name}")
    else:
        print(f"[Comparison] Loading Run 2: {args.run2_method} (auto-selecting latest completed run)")
    results2 = load_run_results(
        data_path, 
        args.run2_method, 
        args.run2_name, 
        auto_run=args.auto_run,
        extra_args=args.run2_extra_args,
        csv_path=csv_path_for_auto_run,
    )
    
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
                    data_path=data_path,
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
                    data_path=data_path,
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
                data_root = ExperimentTracker.compute_data_root(data_path)
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
