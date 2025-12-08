"""Full comparison pipeline: Custom solver vs Commercial solvers.

This script automatically:
1. Creates/uses toy dataset for quick testing
2. Runs custom solver with optimal settings (Adam + BFGS, line search)
3. Runs commercial solvers (pymoo_ga, sklearn)
4. Runs on full dataset
5. Generates comparison reports, graphs, and GIFs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from create_toy_dataset import create_toy_dataset


def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*80}")
    print(f"[Full Comparison] {description}")
    print(f"{'='*80}")
    print(f"[Full Comparison] Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=cwd or Path(__file__).parent,
            capture_output=False,  # Show output in real-time
        )
        print(f"[Full Comparison] ✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Full Comparison] ✗ {description} failed with exit code {e.returncode}")
        return False


def run_custom_solver(
    data_path: Path,
    run_name: Optional[str] = None,
    use_advanced_ga: bool = True,
    extra_args: Optional[List[str]] = None,
    two_stage_optimization: bool = True,
) -> bool:
    """Run custom solver with optimal settings.
    
    If two_stage_optimization is True:
    1. First run Adam to quickly approach optimum
    2. Then run BFGS using Adam's result as initial point for precise optimization
    """
    if two_stage_optimization:
        # Stage 1: Run with Adam (fast initial convergence)
        print(f"\n[Full Comparison] Stage 1: Running Adam optimizer...")
        adam_run_name = f"{run_name}_adam" if run_name else f"adam_{int(time.time())}"
        cmd_adam = [
            sys.executable,
            "main.py",
            "--data-path", str(data_path),
            "--solver-backend", "custom",
            "--solver-method", "adam",
            "--solver-line-search",  # Use line search
            "--solver-second-order", "none",  # No BFGS in first stage
            "--tsne-snapshots",
            "--tsne-interval", "5",
            "--tsne-gif",
            "--tsne-gif-duration", "2.0",
            "--plot-convergence",
        ]
        
        if use_advanced_ga:
            cmd_adam.extend([
                "--use-advanced-ga",
                "--ga-fitness-sharing",
                "--ga-surrogate",
                "--ga-adaptive-population",
                "--ga-island-model",
                "--ga-num-islands", "4",
                "--ga-migration-interval", "10",
            ])
        
        cmd_adam.extend(["--run-name", adam_run_name])
        
        if extra_args:
            cmd_adam.extend(extra_args)
        
        success_adam = run_command(cmd_adam, f"Custom solver (Adam) on {data_path.name}")
        if not success_adam:
            print(f"[Full Comparison] ✗ Adam stage failed, skipping BFGS stage")
            return False
        
        # Wait a moment for file system to sync
        time.sleep(2)
        
        # Stage 2: Run with BFGS (precise optimization using Adam's result as initial point)
        print(f"\n[Full Comparison] Stage 2: Running BFGS optimizer (using Adam result as initial point)...")
        bfgs_run_name = run_name if run_name else f"bfgs_{int(time.time())}"
        cmd_bfgs = [
            sys.executable,
            "main.py",
            "--data-path", str(data_path),
            "--solver-backend", "custom",
            "--solver-method", "adam",  # Still use adam as base, but BFGS will override
            "--solver-second-order", "bfgs",  # Use BFGS for precise optimization
            "--solver-line-search",  # Use line search
            "--reuse-ga-run", adam_run_name,  # Reuse GA results from Adam run
            "--tsne-snapshots",
            "--tsne-interval", "5",
            "--tsne-gif",
            "--tsne-gif-duration", "2.0",
            "--plot-convergence",
        ]
        
        if use_advanced_ga:
            cmd_bfgs.extend([
                "--use-advanced-ga",
                "--ga-fitness-sharing",
                "--ga-surrogate",
                "--ga-adaptive-population",
                "--ga-island-model",
                "--ga-num-islands", "4",
                "--ga-migration-interval", "10",
            ])
        
        cmd_bfgs.extend(["--run-name", bfgs_run_name])
        
        if extra_args:
            cmd_bfgs.extend(extra_args)
        
        success_bfgs = run_command(cmd_bfgs, f"Custom solver (BFGS, initialized from Adam) on {data_path.name}")
        
        if success_bfgs:
            print(f"\n[Full Comparison] ✓ Two-stage optimization completed:")
            print(f"  - Adam run: {adam_run_name}")
            print(f"  - BFGS run: {bfgs_run_name} (final result)")
            return True
        else:
            print(f"\n[Full Comparison] ✗ BFGS stage failed, but Adam result is available: {adam_run_name}")
            return False
    else:
        # Single stage: Use BFGS directly (original behavior)
        cmd = [
            sys.executable,
            "main.py",
            "--data-path", str(data_path),
            "--solver-backend", "custom",
            "--solver-method", "adam",
            "--solver-second-order", "bfgs",
            "--solver-line-search",
            "--tsne-snapshots",
            "--tsne-interval", "5",
            "--tsne-gif",
            "--tsne-gif-duration", "2.0",
            "--plot-convergence",
        ]
        
        if use_advanced_ga:
            cmd.extend([
                "--use-advanced-ga",
                "--ga-fitness-sharing",
                "--ga-surrogate",
                "--ga-adaptive-population",
                "--ga-island-model",
                "--ga-num-islands", "4",
                "--ga-migration-interval", "10",
            ])
        
        if run_name:
            cmd.extend(["--run-name", run_name])
        
        if extra_args:
            cmd.extend(extra_args)
        
        return run_command(cmd, f"Custom solver (BFGS only) on {data_path.name}")


def run_commercial_solver(
    data_path: Path,
    solver: str,
    run_name: Optional[str] = None,
) -> bool:
    """Run a commercial solver."""
    cmd = [
        sys.executable,
        "main_commercial.py",
        "--data-path", str(data_path),
        "--solver", solver,
        "--tsne-snapshots",
        "--tsne-interval", "5",
        "--tsne-gif",
        "--tsne-gif-duration", "2.0",
        "--plot-convergence",
    ]
    
    if run_name:
        cmd.extend(["--run-name", run_name])
    
    solver_name_map = {
        "pymoo_ga": "pymoo GA",
        "sklearn": "sklearn LogisticRegression",
        "gurobi": "Gurobi",
    }
    description = f"Commercial solver ({solver_name_map.get(solver, solver)}) on {data_path.name}"
    
    return run_command(cmd, description)


def run_comparison(
    data_path: Path,
    run1_method: str,
    run2_method: str,
    run1_name: Optional[str] = None,
    run2_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> bool:
    """Run comparison between two solvers."""
    cmd = [
        sys.executable,
        "compare_solvers.py",
        "--data-path", str(data_path),
        "--run1-method", run1_method,
        "--run2-method", run2_method,
        "--plot-convergence",
        "--auto-run",  # Auto-run if results not found
    ]
    
    if run1_name:
        cmd.extend(["--run1-name", run1_name])
    if run2_name:
        cmd.extend(["--run2-name", run2_name])
    if output_name:
        cmd.extend(["--output", output_name])
    
    description = f"Comparison: {run1_method} vs {run2_method}"
    return run_command(cmd, description)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full comparison pipeline: Custom solver vs Commercial solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison with default settings
  python run_full_comparison.py

  # Run only on full dataset (skip toy)
  python run_full_comparison.py --skip-toy

  # Run only on toy dataset
  python run_full_comparison.py --toy-only

  # Skip specific commercial solvers
  python run_full_comparison.py --skip-solvers gurobi

  # Use enhanced GA instead of advanced GA
  python run_full_comparison.py --use-enhanced-ga
  
  # Explicitly use advanced GA (default, but can be specified)
  python run_full_comparison.py --use-advanced-ga
  
  # Disable advanced GA (use basic GA)
  python run_full_comparison.py --no-use-advanced-ga
        """,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../Data"),
        help="Path to data directory (default: ../Data)",
    )
    parser.add_argument(
        "--toy-ratio",
        type=float,
        default=0.1,
        help="Ratio for toy dataset (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--skip-toy",
        action="store_true",
        help="Skip toy dataset comparison",
    )
    parser.add_argument(
        "--toy-only",
        action="store_true",
        help="Run only on toy dataset (skip full dataset)",
    )
    parser.add_argument(
        "--skip-solvers",
        type=str,
        nargs="*",
        default=[],
        help="Skip specific commercial solvers (e.g., --skip-solvers gurobi)",
    )
    parser.add_argument(
        "--use-enhanced-ga",
        action="store_true",
        help="Use enhanced GA instead of advanced GA",
    )
    parser.add_argument(
        "--use-advanced-ga",
        action="store_true",
        help="Use advanced GA (default: True if --use-enhanced-ga is not specified)",
    )
    parser.add_argument(
        "--no-advanced-ga",
        action="store_true",
        help="Disable advanced GA (use basic GA instead)",
    )
    parser.add_argument(
        "--custom-ga-args",
        type=str,
        nargs="*",
        default=None,
        help="Additional arguments for custom solver GA (e.g., --custom-ga-args --ga-population 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Data/comparison_results"),
        help="Directory to save comparison results (default: ../Data/comparison_results)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_dataset_path = data_dir / "creditcard" / "creditcard.csv"
    toy_dataset_path = data_dir / "creditcard" / "creditcard_toy.csv"
    
    # Check if full dataset exists
    if not full_dataset_path.exists():
        print(f"[Full Comparison] ✗ Full dataset not found: {full_dataset_path}")
        print(f"[Full Comparison] Please ensure the dataset exists before running.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"[Full Comparison] Starting Full Comparison Pipeline")
    print(f"{'='*80}")
    print(f"[Full Comparison] Data directory: {data_dir}")
    print(f"[Full Comparison] Output directory: {output_dir}")
    print(f"[Full Comparison] Full dataset: {full_dataset_path}")
    print(f"[Full Comparison] Toy dataset: {toy_dataset_path}")
    print(f"{'='*80}\n")
    
    # Commercial solvers to run
    commercial_solvers = ["pymoo_ga", "sklearn"]
    if "gurobi" not in args.skip_solvers:
        commercial_solvers.append("gurobi")
    
    # Filter out skipped solvers
    commercial_solvers = [s for s in commercial_solvers if s not in args.skip_solvers]
    
    results_summary = []
    
    # ========================================================================
    # Step 1: Create/Use Toy Dataset
    # ========================================================================
    if not args.skip_toy:
        print(f"\n[Full Comparison] Step 1: Creating/Using Toy Dataset")
        if not toy_dataset_path.exists():
            print(f"[Full Comparison] Creating toy dataset from {full_dataset_path.name}...")
            create_toy_dataset(
                input_path=full_dataset_path,
                output_path=toy_dataset_path,
                sample_ratio=args.toy_ratio,
                random_state=42,
                preserve_order=True,
            )
            print(f"[Full Comparison] ✓ Toy dataset created: {toy_dataset_path}")
        else:
            print(f"[Full Comparison] ✓ Toy dataset already exists: {toy_dataset_path}")
        
        # ========================================================================
        # Step 2: Run Custom Solver on Toy Dataset
        # ========================================================================
        print(f"\n[Full Comparison] Step 2: Running Custom Solver on Toy Dataset")
        toy_custom_run_name = f"toy_custom_{int(time.time())}"
        custom_ga_args = args.custom_ga_args or []
        # Determine which GA to use
        # Priority: --use-enhanced-ga > --use-advanced-ga > --no-advanced-ga > default (advanced)
        if args.use_enhanced_ga:
            use_advanced_ga = False
        elif args.no_advanced_ga:
            use_advanced_ga = False
        elif args.use_advanced_ga:
            use_advanced_ga = True
        else:
            # Default: use advanced GA
            use_advanced_ga = True
        
        success = run_custom_solver(
            data_path=toy_dataset_path,
            run_name=toy_custom_run_name,
            use_advanced_ga=use_advanced_ga,
            extra_args=custom_ga_args,
        )
        if success:
            results_summary.append(("Toy Dataset", "Custom Solver", "✓ Success"))
        else:
            results_summary.append(("Toy Dataset", "Custom Solver", "✗ Failed"))
        
        # ========================================================================
        # Step 3: Run Commercial Solvers on Toy Dataset
        # ========================================================================
        print(f"\n[Full Comparison] Step 3: Running Commercial Solvers on Toy Dataset")
        toy_commercial_runs = {}
        for solver in commercial_solvers:
            run_name = f"toy_{solver}_{int(time.time())}"
            success = run_commercial_solver(
                data_path=toy_dataset_path,
                solver=solver,
                run_name=run_name,
            )
            toy_commercial_runs[solver] = run_name if success else None
            if success:
                results_summary.append(("Toy Dataset", f"Commercial {solver}", "✓ Success"))
            else:
                results_summary.append(("Toy Dataset", f"Commercial {solver}", "✗ Failed"))
        
        # ========================================================================
        # Step 4: Compare Custom vs Commercial on Toy Dataset
        # ========================================================================
        print(f"\n[Full Comparison] Step 4: Comparing Solvers on Toy Dataset")
        for solver in commercial_solvers:
            if toy_commercial_runs.get(solver):
                output_path = output_dir / f"toy_custom_vs_{solver}.txt"
                success = run_comparison(
                    data_path=toy_dataset_path,
                    run1_method="without_hessian",
                    run2_method=f"commercial_{solver}",
                    run1_name=toy_custom_run_name,
                    run2_name=toy_commercial_runs[solver],
                    output_name=str(output_path),
                )
                if success:
                    results_summary.append(("Toy Comparison", f"Custom vs {solver}", "✓ Success"))
                else:
                    results_summary.append(("Toy Comparison", f"Custom vs {solver}", "✗ Failed"))
    
    # ========================================================================
    # Step 5: Run on Full Dataset (if not toy-only)
    # ========================================================================
    if not args.toy_only:
        print(f"\n[Full Comparison] Step 5: Running Custom Solver on Full Dataset")
        full_custom_run_name = f"full_custom_{int(time.time())}"
        custom_ga_args = args.custom_ga_args or []
        # Determine which GA to use (same logic as toy dataset)
        if args.use_enhanced_ga:
            use_advanced_ga = False
        elif args.no_advanced_ga:
            use_advanced_ga = False
        elif args.use_advanced_ga:
            use_advanced_ga = True
        else:
            # Default: use advanced GA
            use_advanced_ga = True
        
        success = run_custom_solver(
            data_path=full_dataset_path,
            run_name=full_custom_run_name,
            use_advanced_ga=use_advanced_ga,
            extra_args=custom_ga_args,
        )
        if success:
            results_summary.append(("Full Dataset", "Custom Solver", "✓ Success"))
        else:
            results_summary.append(("Full Dataset", "Custom Solver", "✗ Failed"))
        
        # ========================================================================
        # Step 6: Run Commercial Solvers on Full Dataset
        # ========================================================================
        print(f"\n[Full Comparison] Step 6: Running Commercial Solvers on Full Dataset")
        full_commercial_runs = {}
        for solver in commercial_solvers:
            run_name = f"full_{solver}_{int(time.time())}"
            success = run_commercial_solver(
                data_path=full_dataset_path,
                solver=solver,
                run_name=run_name,
            )
            full_commercial_runs[solver] = run_name if success else None
            if success:
                results_summary.append(("Full Dataset", f"Commercial {solver}", "✓ Success"))
            else:
                results_summary.append(("Full Dataset", f"Commercial {solver}", "✗ Failed"))
        
        # ========================================================================
        # Step 7: Compare Custom vs Commercial on Full Dataset
        # ========================================================================
        print(f"\n[Full Comparison] Step 7: Comparing Solvers on Full Dataset")
        for solver in commercial_solvers:
            if full_commercial_runs.get(solver):
                output_path = output_dir / f"full_custom_vs_{solver}.txt"
                success = run_comparison(
                    data_path=full_dataset_path,
                    run1_method="without_hessian",
                    run2_method=f"commercial_{solver}",
                    run1_name=full_custom_run_name,
                    run2_name=full_commercial_runs[solver],
                    output_name=str(output_path),
                )
                if success:
                    results_summary.append(("Full Comparison", f"Custom vs {solver}", "✓ Success"))
                else:
                    results_summary.append(("Full Comparison", f"Custom vs {solver}", "✗ Failed"))
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"[Full Comparison] Pipeline Summary")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Solver':<30} {'Status':<10}")
    print(f"{'-'*80}")
    for dataset, solver, status in results_summary:
        print(f"{dataset:<20} {solver:<30} {status:<10}")
    print(f"{'='*80}")
    
    # Save summary to file
    summary_path = output_dir / "pipeline_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Full Comparison Pipeline Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Dataset':<20} {'Solver':<30} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        for dataset, solver, status in results_summary:
            f.write(f"{dataset:<20} {solver:<30} {status:<10}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n[Full Comparison] Summary saved to: {summary_path}")
    print(f"[Full Comparison] All comparison results saved to: {output_dir}")
    print(f"\n[Full Comparison] ✓ Pipeline completed!")


if __name__ == "__main__":
    main()

