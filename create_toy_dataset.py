"""Create a toy dataset for testing compare_solvers.py.

This script creates a smaller version of the credit card fraud dataset
by randomly sampling 10% of the rows while preserving the order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def create_toy_dataset(
    input_path: Path,
    output_path: Path,
    sample_ratio: float = 0.1,
    random_state: int = 42,
    preserve_order: bool = True,
) -> None:
    """Create a toy dataset by sampling a percentage of rows.
    
    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    output_path : Path
        Path to save the output CSV file.
    sample_ratio : float, default=0.1
        Fraction of rows to sample (0.1 = 10%).
    random_state : int, default=42
        Random seed for reproducibility.
    preserve_order : bool, default=True
        If True, preserves the original row order. If False, shuffles.
    """
    print(f"[Toy Dataset] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    target_rows = int(original_rows * sample_ratio)
    
    print(f"[Toy Dataset] Original dataset: {original_rows:,} rows")
    print(f"[Toy Dataset] Target sample size: {target_rows:,} rows ({sample_ratio*100:.1f}%)")
    
    if preserve_order:
        # Sample rows while preserving order
        # Use systematic sampling: take every nth row
        step = int(1 / sample_ratio)
        sampled_indices = list(range(0, original_rows, step))[:target_rows]
        df_toy = df.iloc[sampled_indices].copy()
        print(f"[Toy Dataset] Systematic sampling: every {step}th row")
    else:
        # Random sampling
        df_toy = df.sample(n=target_rows, random_state=random_state).copy()
        df_toy = df_toy.sort_index()  # Sort by original index to preserve some order
        print(f"[Toy Dataset] Random sampling with seed={random_state}")
    
    # Reset index to start from 0
    df_toy = df_toy.reset_index(drop=True)
    
    # Check class distribution
    if "Class" in df_toy.columns:
        fraud_count = df_toy["Class"].sum()
        total_count = len(df_toy)
        fraud_ratio = fraud_count / total_count if total_count > 0 else 0
        
        print(f"[Toy Dataset] Class distribution:")
        print(f"  Total samples: {total_count:,}")
        print(f"  Fraud cases: {fraud_count:,} ({fraud_ratio*100:.2f}%)")
        print(f"  Benign cases: {total_count - fraud_count:,} ({(1-fraud_ratio)*100:.2f}%)")
    
    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_toy.to_csv(output_path, index=False)
    
    print(f"[Toy Dataset] Toy dataset saved to: {output_path}")
    print(f"[Toy Dataset] Final dataset size: {len(df_toy):,} rows, {len(df_toy.columns)} columns")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a toy dataset for testing compare_solvers.py"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Data/creditcard/creditcard.csv"),
        help="Path to the input CSV file (default: ../Data/creditcard/creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Data/creditcard/creditcard_toy.csv"),
        help="Path to save the output CSV file (default: ../Data/creditcard/creditcard_toy.csv)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows to sample (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before sampling (default: preserves order with systematic sampling)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[Error] Input file not found: {args.input}")
        return
    
    create_toy_dataset(
        input_path=args.input,
        output_path=args.output,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
        preserve_order=not args.shuffle,
    )
    
    print("\n[Toy Dataset] Usage example:")
    print(f"  python compare_solvers.py \\")
    print(f"    --data-path {args.output} \\")
    print(f"    --run1-method without_hessian \\")
    print(f"    --run2-method commercial_sklearn")


if __name__ == "__main__":
    main()

