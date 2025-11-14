"""Generate t-SNE visualization of actual fraud/benign labels (no predictions)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_actual_labels_tsne(
    data_path: Path,
    target_column: str = "Class",
    sample_size: int = 5000,
    output_path: Path | None = None,
) -> Path:
    """Generate t-SNE visualization showing only actual fraud/benign labels.
    
    Parameters
    ----------
    data_path : Path
        Path to the credit card fraud dataset CSV file.
    target_column : str, default="Class"
        Name of the target column in the dataset.
    sample_size : int, default=5000
        Maximum number of samples to use for t-SNE (for performance).
    output_path : Path, optional
        Output path for the PNG file. If None, saves to Data/tsne_actual_labels.png.
    
    Returns
    -------
    Path
        Path to the generated PNG file.
    """
    # Load original dataset directly
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"[t-SNE] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    
    print(f"[t-SNE] Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"[t-SNE] Fraud cases: {y.sum()}, Benign cases: {(y == 0).sum()}")
    
    # Sample if needed
    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        sample_indices = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        X_used = X.iloc[sample_indices].reset_index(drop=True)
        y_used = y.iloc[sample_indices].reset_index(drop=True)
        print(f"[t-SNE] Sampling {sample_size} samples for visualization")
    else:
        X_used = X.reset_index(drop=True)
        y_used = y.reset_index(drop=True)
    
    # Compute t-SNE embedding
    print(f"[t-SNE] Computing t-SNE embedding on {len(X_used)} samples...")
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", n_iter=1000)
    embedding = tsne.fit_transform(X_used.to_numpy(dtype=float))
    print("[t-SNE] Embedding completed")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_np = y_used.to_numpy(dtype=int)
    fraud_mask = y_np == 1
    benign_mask = y_np == 0
    
    # Plot benign cases (blue)
    if benign_mask.any():
        ax.scatter(
            embedding[benign_mask, 0],
            embedding[benign_mask, 1],
            c="#1f77b4",  # Blue
            s=20,
            alpha=0.6,
            edgecolors="none",
            label="Actual Benign",
            zorder=1,
        )
    
    # Plot fraud cases (red, larger and more visible)
    if fraud_mask.any():
        ax.scatter(
            embedding[fraud_mask, 0],
            embedding[fraud_mask, 1],
            c="#d62728",  # Red
            s=40,
            alpha=0.8,
            edgecolors="#000000",
            linewidths=0.5,
            label="Actual Fraud",
            zorder=2,
        )
    
    # Create legend
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", markersize=8, label="Actual Benign", alpha=0.6),
        Line2D([], [], marker="o", linestyle="", color="#d62728", markersize=10, markeredgecolor="#000000", markeredgewidth=0.5, label="Actual Fraud", alpha=0.8),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    
    ax.set_title("t-SNE Visualization of Actual Labels", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE Component 1", fontsize=10)
    ax.set_ylabel("t-SNE Component 2", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    
    # Save to Data folder
    if output_path is None:
        # Save to the same directory as the data file
        output_path = data_path.parent / "tsne_actual_labels.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[t-SNE] Visualization saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualization of actual fraud/benign labels (no predictions)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../Data/creditcard/creditcard.csv"),
        help="Path to the credit card fraud dataset CSV file (default: ../Data/creditcard/creditcard.csv).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Class",
        help="Name of the target column in the dataset (default: Class).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Maximum number of samples to use for t-SNE (default: 5000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the PNG file. If not specified, saves to Data/tsne_actual_labels.png",
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_actual_labels_tsne(
            data_path=args.data_path,
            target_column=args.target_column,
            sample_size=args.sample_size,
            output_path=args.output,
        )
        print(f"\n✓ Successfully generated visualization: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

