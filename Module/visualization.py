"""Visualization utilities for solver optimization tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from scipy.optimize import minimize_scalar
from sklearn.metrics import fbeta_score


def generate_tsne_snapshots(
    X: pd.DataFrame,
    y: pd.Series,
    snapshots: Sequence[Dict[str, object]],
    output_root: Path,
    threshold: Optional[float] = None,
    use_adaptive_threshold: bool = True,
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

    # Calculate loss for each snapshot to show in title
    from Solver.cost_functions import cost_sensitive_nll
    
    # Track statistics across snapshots for debugging
    prev_weights_norm = None
    
    for snapshot in snapshots:
        iteration = int(snapshot["iteration"])
        weights = np.asarray(snapshot["weights"], dtype=float)
        bias = float(snapshot["bias"])

        # Debug: Check if weights are changing
        weights_norm = np.linalg.norm(weights)
        weights_change = abs(weights_norm - prev_weights_norm) if prev_weights_norm is not None else 0.0
        prev_weights_norm = weights_norm

        logits = X_used.to_numpy(dtype=float) @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # Clip to avoid overflow
        
        # Use adaptive threshold for each snapshot if requested
        if use_adaptive_threshold and threshold is None:
            # Optimize threshold for this specific snapshot's model
            def objective(t: float) -> float:
                preds_t = (probs >= t).astype(int)
                # Use default beta=2.0 for F-beta score
                score = fbeta_score(y_used, preds_t, beta=2.0, zero_division=0)
                return -score
            
            try:
                result = minimize_scalar(objective, bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-3})
                snapshot_threshold = float(result.x)
            except Exception:
                snapshot_threshold = 0.5  # Fallback
        else:
            snapshot_threshold = threshold if threshold is not None else 0.5
        
        preds = (probs >= snapshot_threshold).astype(int)
        
        # Debug info
        prob_mean = float(np.mean(probs))
        prob_std = float(np.std(probs))
        pred_fraud_count = int(np.sum(preds))
        
        # Print debug info to console
        if iteration == 0 or iteration % 10 == 0 or weights_change > 1e-6:
            print(
                f"[t-SNE Debug] Iter {iteration:4d} | "
                f"|W|={weights_norm:.4f} | "
                f"ΔW={weights_change:.6f} | "
                f"P_mean={prob_mean:.4f} | "
                f"P_std={prob_std:.4f} | "
                f"Pred_fraud={pred_fraud_count:4d}"
            )
        
        # Calculate loss for this snapshot (using training data weights if available)
        # We'll use a simple sample weight for visualization
        sample_weights_viz = np.ones(len(y_used))
        sample_weights_viz[y_used == 1] = 5.0  # Default cost_beta
        try:
            loss = cost_sensitive_nll(weights, bias, X_used.to_numpy(dtype=float), y_used.to_numpy(dtype=float), sample_weights_viz)
        except Exception:
            loss = float('nan')

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use probability as color intensity: red for high prob (fraud), blue for low prob (legit)
        # Create a colormap: blue (0.0) -> white (0.5) -> red (1.0)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#1f77b4', '#ffffff', '#d62728']  # blue -> white -> red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('prob_map', colors_list, N=n_bins)
        
        # Scatter plot with probability-based colors
        scatter = ax.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=probs, 
            cmap=cmap, 
            vmin=0, 
            vmax=1,
            s=25, 
            alpha=0.7, 
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Fraud Probability', shrink=0.8)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Overlay predicted fraud (above threshold) with bright yellow edge - draw first so it's visible
        pred_fraud_mask = preds == 1
        if pred_fraud_mask.any():
            ax.scatter(
                embedding[pred_fraud_mask, 0],
                embedding[pred_fraud_mask, 1],
                facecolors="none",
                edgecolors="#FFD700",  # Gold color, more visible
                s=70,
                linewidths=2.0,  # Thicker line
                label="Predicted Fraud",
                zorder=11,  # Above actual fraud
                linestyle='-',
            )
        
        # Overlay actual fraud cases with thinner, semi-transparent outline
        positive_mask = y_np == 1
        if positive_mask.any():
            # Use diamond marker for actual fraud to distinguish from predicted
            ax.scatter(
                embedding[positive_mask, 0],
                embedding[positive_mask, 1],
                facecolors="none",
                edgecolors="#333333",  # Dark gray instead of pure black
                s=100,
                linewidths=0.8,  # Thinner line
                label="Actual Fraud",
                zorder=10,  # Below predicted fraud
                marker='D',  # Diamond marker
                alpha=0.7,  # Semi-transparent
            )

        # Create legend
        handles = [
            Line2D([], [], marker="o", linestyle="", color="#1f77b4", markersize=8, label="Low Prob (Legit)"),
            Line2D([], [], marker="o", linestyle="", color="#d62728", markersize=8, label="High Prob (Fraud)"),
        ]
        if pred_fraud_mask.any():
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="-",
                    markerfacecolor="none",
                    markeredgecolor="#FFD700",
                    markersize=10,
                    markeredgewidth=2.0,
                    label="Predicted Fraud",
                )
            )
        if positive_mask.any():
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="D",  # Diamond marker
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor="#333333",
                    markersize=8,
                    markeredgewidth=0.8,
                    label="Actual Fraud",
                )
            )
        
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
        
        # Title with iteration, loss, and debug info
        loss_str = f"Loss={loss:.4f}" if not np.isnan(loss) else "Loss=N/A"
        threshold_str = f"T={snapshot_threshold:.3f}" if use_adaptive_threshold else f"T={threshold:.3f}"
        debug_str = f"|W|={weights_norm:.3f}"
        if weights_change > 0:
            debug_str += f" Δ={weights_change:.4f}"
        debug_str += f" P_mean={prob_mean:.3f} Pred_fraud={pred_fraud_count} {threshold_str}"
        ax.set_title(f"t-SNE Snapshot (iter {iteration}, {loss_str})\n{debug_str}", fontsize=9, fontweight='bold')
        ax.set_xlabel("t-SNE Component 1", fontsize=9)
        ax.set_ylabel("t-SNE Component 2", fontsize=9)
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


def generate_single_model_tsne(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    bias: float,
    output_root: Path,
    threshold: Optional[float] = None,
    use_adaptive_threshold: bool = True,
    title_suffix: str = "",
) -> Path:
    """Generate a single t-SNE visualization for a trained model (no iteration snapshots).
    
    This function creates a scatter plot showing the model's predictions on the data,
    with points colored by predicted probability and outlined for actual/predicted fraud.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for visualization (typically training data).
    y : pd.Series
        True labels (0 = legitimate, 1 = fraud).
    weights : np.ndarray
        Model weights vector.
    bias : float
        Model bias term.
    output_root : Path
        Root directory where the image will be saved.
    threshold : float, optional
        Decision threshold for converting probabilities to binary predictions.
        If None and use_adaptive_threshold is True, will be optimized.
    use_adaptive_threshold : bool, default=True
        If True, optimize threshold for this model.
    title_suffix : str, default=""
        Additional text to append to the plot title.
    
    Returns
    -------
    Path
        Path to the generated PNG file.
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        print("[t-SNE] Not enough data points or features to compute embedding.")
        return None
    
    sample_size = min(len(X), 5000)
    if sample_size < len(X):
        rng = np.random.default_rng(42)
        sample_indices = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        X_used = X.iloc[sample_indices].reset_index(drop=True)
        y_used = y.iloc[sample_indices].reset_index(drop=True)
    else:
        X_used = X.reset_index(drop=True)
        y_used = y.reset_index(drop=True)
    
    print(f"[t-SNE] Generating embedding on {len(X_used)} samples.")
    
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", max_iter=1000)
    embedding = tsne.fit_transform(X_used.to_numpy(dtype=float))
    
    snapshot_dir = output_root / "tsne_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    y_np = y_used.to_numpy(dtype=int)
    
    # Compute predictions
    logits = X_used.to_numpy(dtype=float) @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    
    # Optimize threshold if needed
    if use_adaptive_threshold and threshold is None:
        def objective(t: float) -> float:
            preds_t = (probs >= t).astype(int)
            score = fbeta_score(y_used, preds_t, beta=2.0, zero_division=0)
            return -score
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-3})
            snapshot_threshold = float(result.x)
        except Exception:
            snapshot_threshold = 0.5
    else:
        snapshot_threshold = threshold if threshold is not None else 0.5
    
    preds = (probs >= snapshot_threshold).astype(int)
    
    # Calculate loss
    from Solver.cost_functions import cost_sensitive_nll
    sample_weights_viz = np.ones(len(y_used))
    sample_weights_viz[y_used == 1] = 5.0
    try:
        loss = cost_sensitive_nll(weights, bias, X_used.to_numpy(dtype=float), y_used.to_numpy(dtype=float), sample_weights_viz)
    except Exception:
        loss = float('nan')
    
    weights_norm = np.linalg.norm(weights)
    prob_mean = float(np.mean(probs))
    pred_fraud_count = int(np.sum(preds))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use probability as color intensity
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#1f77b4', '#ffffff', '#d62728']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('prob_map', colors_list, N=n_bins)
    
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=probs,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=25,
        alpha=0.7,
        edgecolors='none'
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Fraud Probability', shrink=0.8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Overlay predicted fraud
    pred_fraud_mask = preds == 1
    if pred_fraud_mask.any():
        ax.scatter(
            embedding[pred_fraud_mask, 0],
            embedding[pred_fraud_mask, 1],
            facecolors="none",
            edgecolors="#FFD700",
            s=70,
            linewidths=2.0,
            label="Predicted Fraud",
            zorder=11,
        )
    
    # Overlay actual fraud
    positive_mask = y_np == 1
    if positive_mask.any():
        ax.scatter(
            embedding[positive_mask, 0],
            embedding[positive_mask, 1],
            facecolors="none",
            edgecolors="#333333",
            s=100,
            linewidths=0.8,
            label="Actual Fraud",
            zorder=10,
            marker='D',
            alpha=0.7,
        )
    
    # Create legend
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", markersize=8, label="Low Prob (Legit)"),
        Line2D([], [], marker="o", linestyle="", color="#d62728", markersize=8, label="High Prob (Fraud)"),
    ]
    if pred_fraud_mask.any():
        handles.append(
            Line2D([], [], marker="o", linestyle="-", markerfacecolor="none", markeredgecolor="#FFD700", markersize=10, markeredgewidth=2.0, label="Predicted Fraud")
        )
    if positive_mask.any():
        handles.append(
            Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="#333333", markersize=8, markeredgewidth=0.8, label="Actual Fraud")
        )
    
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    
    # Title
    loss_str = f"Loss={loss:.4f}" if not np.isnan(loss) else "Loss=N/A"
    threshold_str = f"T={snapshot_threshold:.3f}"
    debug_str = f"|W|={weights_norm:.3f} P_mean={prob_mean:.3f} Pred_fraud={pred_fraud_count} {threshold_str}"
    title = f"t-SNE Visualization ({loss_str}){title_suffix}\n{debug_str}"
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel("t-SNE Component 1", fontsize=9)
    ax.set_ylabel("t-SNE Component 2", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    
    image_path = snapshot_dir / "tsne_final_model.png"
    fig.savefig(image_path, dpi=220)
    plt.close(fig)
    
    print(f"[t-SNE] Visualization saved to: {image_path}")
    return image_path


__all__ = ["generate_tsne_snapshots", "generate_single_model_tsne"]

