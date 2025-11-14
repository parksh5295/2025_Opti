"""Visualization utilities for solver optimization tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


def generate_tsne_snapshots(
    X: pd.DataFrame,
    y: pd.Series,
    snapshots: Sequence[Dict[str, object]],
    output_root: Path,
    threshold: float,
    gif: bool = False,
    gif_duration: float = 0.6,
) -> List[Path]:
    """Generate t-SNE visualizations of solver predictions at different iterations.
    
    This function creates scatter plots showing how the solver's predictions
    evolve over iterations, with points colored by predicted class (red for fraud,
    blue for legitimate) and outlined if they are actually fraud cases.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for visualization (typically training or test data).
    y : pd.Series
        True labels (0 = legitimate, 1 = fraud).
    snapshots : Sequence[Dict[str, object]]
        List of solver parameter snapshots, each containing 'iteration', 'weights', and 'bias'.
    output_root : Path
        Root directory where snapshot images will be saved.
    threshold : float
        Decision threshold for converting probabilities to binary predictions.
    gif : bool, default=False
        If True, combine all snapshots into an animated GIF.
    gif_duration : float, default=0.6
        Frame duration in seconds for the animated GIF.
    
    Returns
    -------
    List[Path]
        List of paths to generated image files (and GIF if requested).
    """
    if not snapshots:
        return []
    if X.shape[0] < 2 or X.shape[1] < 2:
        print("[t-SNE] Not enough data points or features to compute embedding.")
        return []

    sample_size = min(len(X), 5000)
    if sample_size < len(X):
        rng = np.random.default_rng(0)
        sample_indices = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        X_used = X.iloc[sample_indices].reset_index(drop=True)
        y_used = y.iloc[sample_indices].reset_index(drop=True)
    else:
        X_used = X.reset_index(drop=True)
        y_used = y.reset_index(drop=True)

    print(f"[t-SNE] Generating embeddings on {len(X_used)} samples.")

    tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(X_used.to_numpy(dtype=float))

    snapshot_dir = output_root / "tsne_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = []

    y_np = y_used.to_numpy(dtype=int)

    for snapshot in snapshots:
        iteration = int(snapshot["iteration"])
        weights = np.asarray(snapshot["weights"], dtype=float)
        bias = float(snapshot["bias"])

        logits = X_used.to_numpy(dtype=float) @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = np.where(preds == 1, "#d62728", "#1f77b4")
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=18, alpha=0.65, edgecolors="none")

        positive_mask = y_np == 1
        if positive_mask.any():
            ax.scatter(
                embedding[positive_mask, 0],
                embedding[positive_mask, 1],
                facecolors="none",
                edgecolors="#111111",
                s=60,
                linewidths=0.8,
                label="Actual Fraud",
            )

        handles = [
            Line2D([], [], marker="o", linestyle="", color="#d62728", label="Predicted Fraud"),
            Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="Predicted Legit"),
        ]
        if positive_mask.any():
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor="#111111",
                    label="Actual Fraud",
                )
            )

        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_title(f"t-SNE Snapshot (iter {iteration})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)
        fig.tight_layout()

        image_path = snapshot_dir / f"snapshot_iter_{iteration:04d}.png"
        fig.savefig(image_path, dpi=220)
        plt.close(fig)

        image_paths.append(image_path)

    if gif and image_paths:
        gif_path = snapshot_dir / "tsne_evolution.gif"
        frames = [imageio.imread(path) for path in image_paths]
        imageio.mimsave(gif_path, frames, duration=gif_duration)
        image_paths.append(gif_path)

    return image_paths


__all__ = ["generate_tsne_snapshots"]

