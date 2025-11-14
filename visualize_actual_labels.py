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
    run_name: str,
    method_label: str = "without_hessian",
    sample_size: int = 5000,
    output_path: Path | None = None,
) -> Path:
    """Generate t-SNE visualization showing only actual fraud/benign labels.
    
    Parameters
    ----------
    data_path : Path
        Path to the data directory (parent of Result/ and log/).
    run_name : str
        Run name to load preprocessing data from.
    method_label : str, default="without_hessian"
        Method label (with_hessian or without_hessian).
    sample_size : int, default=5000
        Maximum number of samples to use for t-SNE (for performance).
    output_path : Path, optional
        Output path for the PNG file. If None, saves to Result/{method_label}/{run_name}/.
    
    Returns
    -------
    Path
        Path to the generated PNG file.
    """
    from Module import ExperimentTracker, load_and_preprocess, PreprocessingConfig
    
    # Load preprocessing data from the specified run
    base_root = ExperimentTracker.compute_data_root(data_path)
    log_root = base_root / "log" / method_label
    state_path = log_root / run_name / "state.json"
    
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    
    import json
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    
    if "preprocessing" not in state.get("stages", {}):
        raise RuntimeError("Preprocessing stage not found in the specified run.")
    
    result_dir = Path(state["result_dir"])
    preproc_stage = state["stages"]["preprocessing"]
    preproc_artifacts = preproc_stage["artifacts"]
    
    # Load training data (resampled)
    X_train_path = result_dir / preproc_artifacts["X_train_res"]["path"]
    y_train_path = result_dir / preproc_artifacts["y_train_res"]["path"]
    
    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found in {result_dir}")
    
    X_train = pd.read_pickle(X_train_path)
    y_train = pd.read_pickle(y_train_path)
    
    print(f"[t-SNE] Loaded {len(X_train)} samples with {len(X_train.columns)} features")
    print(f"[t-SNE] Fraud cases: {y_train.sum()}, Benign cases: {(y_train == 0).sum()}")
    
    # Sample if needed
    if len(X_train) > sample_size:
        rng = np.random.default_rng(42)
        sample_indices = np.sort(rng.choice(len(X_train), size=sample_size, replace=False))
        X_used = X_train.iloc[sample_indices].reset_index(drop=True)
        y_used = y_train.iloc[sample_indices].reset_index(drop=True)
        print(f"[t-SNE] Sampling {sample_size} samples for visualization")
    else:
        X_used = X_train.reset_index(drop=True)
        y_used = y_train.reset_index(drop=True)
    
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
    
    ax.set_title(f"t-SNE Visualization of Actual Labels\n(Run: {run_name}, Method: {method_label})", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE Component 1", fontsize=10)
    ax.set_ylabel("t-SNE Component 2", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    
    # Save
    if output_path is None:
        output_path = result_dir / "tsne_actual_labels.png"
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
        required=True,
        help="Path to the data directory (parent of Result/ and log/). Can be the CSV file path or data root.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run name to load preprocessing data from.",
    )
    parser.add_argument(
        "--method-label",
        type=str,
        default="without_hessian",
        choices=["without_hessian", "with_hessian"],
        help="Method label to search for the run.",
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
        help="Output path for the PNG file. If not specified, saves to Result/{method_label}/{run_name}/tsne_actual_labels.png",
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_actual_labels_tsne(
            data_path=args.data_path,
            run_name=args.run_name,
            method_label=args.method_label,
            sample_size=args.sample_size,
            output_path=args.output,
        )
        print(f"\n✓ Successfully generated visualization: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

