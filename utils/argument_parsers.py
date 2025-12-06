"""Argument parser definitions for main pipeline scripts.

This module contains the argument parser for main.py to reduce its length.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def create_main_parser() -> argparse.ArgumentParser:
    """Create argument parser for main.py pipeline."""
    parser = argparse.ArgumentParser(description="Information-theoretic ensemble fraud detection pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the credit card fraud dataset (CSV).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional identifier used for result/log directory names.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a previous run directory or state.json file to resume from.",
    )
    parser.add_argument(
        "--force-new-run",
        action="store_true",
        help="Ignore existing states and always start a new run.",
    )
    parser.add_argument(
        "--feature-ensemble-mode",
        type=str,
        default="scores",
        choices=["scores", "construct"],
        help="How to aggregate feature selection methods: 'scores' uses weighted importance scores only, 'construct' also creates new aggregate features.",
    )
    parser.add_argument(
        "--feature-ensemble-top-k",
        type=int,
        default=5,
        help="Number of top-ranked features per method to use when constructing ensemble features.",
    )
    parser.add_argument(
        "--reuse-ga-run",
        type=str,
        default=None,
        help="Reuse GA results from the specified run name instead of re-running GA.",
    )
    parser.add_argument(
        "--reuse-ga-method",
        type=str,
        default=None,
        help="Method label (with_hessian / without_hessian) to use when loading GA results. Defaults to current method.",
    )
    parser.add_argument("--target-column", type=str, default="Class", help="Name of the target column in the dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion reserved for the test split.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation split fraction from the training pool.")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for F-beta and cost-sensitive metrics.")
    parser.add_argument("--cost-beta", type=float, default=5.0, help="Sample weight multiplier for the fraud class.")
    parser.add_argument("--weight-pca", type=float, default=0.35, help="Ensemble weight for PCA-based importance.")
    parser.add_argument("--weight-mi", type=float, default=0.35, help="Ensemble weight for mutual information scores.")
    parser.add_argument("--weight-rf", type=float, default=0.30, help="Ensemble weight for random forest importances.")
    parser.add_argument("--penalty-weight-cmi", type=float, default=0.6, help="Weight assigned to conditional mutual information in the redundancy penalty.")
    parser.add_argument("--penalty-weight-corr", type=float, default=0.25, help="Weight assigned to Pearson correlation in the redundancy penalty.")
    parser.add_argument("--penalty-weight-vif", type=float, default=0.15, help="Weight assigned to VIF contributions in the redundancy penalty.")
    parser.add_argument("--redundancy-budget", type=float, default=0.75, help="Maximum cumulative redundancy penalty before rejecting a feature during selection.")
    parser.add_argument("--lambda-penalty", type=float, default=0.05, help="Regularisation strength on redundancy penalty in GA fitness and evaluation.")
    parser.add_argument("--alpha-size", type=float, default=0.01, help="Penalty applied per selected feature in GA fitness and evaluation.")
    parser.add_argument("--min-candidate-features", type=int, default=12, help="Minimum number of features retained after redundancy screening.")
    parser.add_argument("--ga-population", type=int, default=60, help="Population size for the genetic algorithm.")
    parser.add_argument("--ga-generations", type=int, default=40, help="Number of GA generations.")
    parser.add_argument("--ga-mutation", type=float, default=0.08, help="Mutation probability for each gene in GA.")
    parser.add_argument("--ga-min-features", type=int, default=6, help="Minimum number of active features enforced in GA individuals.")
    parser.add_argument("--ga-cv-splits", type=int, default=5, help="Number of CV folds inside GA fitness evaluation.")
    parser.add_argument("--ga-quiet", action="store_true", help="Suppress per-generation GA logs.")
    parser.add_argument(
        "--use-enhanced-ga",
        action="store_true",
        help="Use enhanced GA with advanced techniques (heuristic init, adaptive mutation, local search, etc.)",
    )
    parser.add_argument(
        "--ga-crossover-type",
        type=str,
        default="uniform",
        choices=["single", "two_point", "uniform"],
        help="Crossover strategy for enhanced GA (default: uniform).",
    )
    parser.add_argument(
        "--ga-local-search",
        action="store_true",
        default=True,
        help="Enable local search (hill climbing) in enhanced GA (default: True).",
    )
    parser.add_argument(
        "--ga-early-stopping",
        type=int,
        default=15,
        help="Early stopping patience (generations without improvement) for enhanced GA (0 to disable).",
    )
    parser.add_argument(
        "--ga-heuristic-init",
        type=float,
        default=0.3,
        help="Ratio of population initialized using feature importance in enhanced GA (0.0 to 1.0).",
    )
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    parser.add_argument("--enable-undersampling", action="store_true", help="Enable random undersampling before SMOTE.")
    parser.add_argument("--undersample-ratio", type=float, default=None, help="Majority/minority ratio for undersampling (if enabled).")
    parser.add_argument("--rho-auc", type=float, default=0.35, help="Weight for ROC-AUC in the overall evaluation score.")
    parser.add_argument("--rho-f1", type=float, default=0.25, help="Weight for cost-sensitive F1 in the overall evaluation score.")
    parser.add_argument("--rho-pr", type=float, default=0.2, help="Weight for PR-AUC in the overall evaluation score.")
    parser.add_argument("--rho-gmean", type=float, default=0.1, help="Weight for G-mean in the overall evaluation score.")
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="custom",
        choices=["custom", "sklearn"],
        help="Backend used for optimising logistic parameters (default: custom solver).",
    )
    parser.add_argument("--solver-max-iter", type=int, default=400, help="Maximum iterations for the custom solver.")
    parser.add_argument("--solver-lr", type=float, default=0.1, help="Learning rate for the custom solver.")
    parser.add_argument("--solver-tol", type=float, default=1e-5, help="Tolerance for solver convergence.")
    parser.add_argument("--solver-momentum", type=float, default=0.0, help="Momentum term for solver gradient descent.")
    parser.add_argument(
        "--solver-method",
        type=str,
        default="gd",
        choices=["gd", "momentum", "nesterov", "adam"],
        help="First-order optimisation variant to use.",
    )
    parser.add_argument("--solver-line-search", action="store_true", help="Enable backtracking line search.")
    parser.add_argument("--solver-line-alpha", type=float, default=0.3, help="Armijo alpha parameter for line search.")
    parser.add_argument("--solver-line-beta", type=float, default=0.8, help="Backtracking decay factor for line search.")
    parser.add_argument("--solver-adam-beta1", type=float, default=0.9, help="Adam beta1 parameter.")
    parser.add_argument("--solver-adam-beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--solver-adam-epsilon", type=float, default=1e-8, help="Adam epsilon parameter.")
    parser.add_argument(
        "--solver-second-order",
        type=str,
        default="none",
        choices=["none", "bfgs"],
        help="Enable quasi-Newton optimisation (BFGS).",
    )
    parser.add_argument("--solver-verbose", action="store_true", help="Print solver progress information.")
    parser.add_argument("--solver-track-history", action="store_true", help="Record loss history during solver optimisation.")
    parser.add_argument(
        "--plot-convergence",
        action="store_true",
        help="Generate convergence plots for GA and/or Solver optimization history.",
    )
    parser.add_argument(
        "--tsne-snapshots",
        action="store_true",
        help="Generate t-SNE visualisations of solver predictions at snapshot intervals (custom solver only).",
    )
    parser.add_argument(
        "--tsne-interval",
        type=int,
        default=5,
        help="Iteration interval used when capturing solver parameter snapshots (custom solver only).",
    )
    parser.add_argument(
        "--tsne-gif",
        action="store_true",
        help="Combine generated t-SNE snapshots into an animated GIF.",
    )
    parser.add_argument(
        "--tsne-gif-duration",
        type=float,
        default=2.0,
        help="Frame duration, in seconds, for the animated t-SNE GIF (default: 2.0).",
    )
    parser.add_argument("--skip-explainability", action="store_true", help="Skip SHAP-based explainability step.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used across the pipeline.")
    return parser

