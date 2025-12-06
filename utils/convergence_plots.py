"""Convergence plotting utilities for GA and Solver optimization tracking.

This module provides functions to visualize the convergence of:
- GA: Generation vs. Best Fitness Score
- Solver: Iteration vs. Loss
- Enhanced GA: Generation vs. Best Fitness + Diversity
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_ga_convergence(
    history: List[float],
    output_path: Path,
    title: str = "GA Convergence",
    xlabel: str = "Generation",
    ylabel: str = "Best Fitness Score",
    show_improvement: bool = True,
) -> Path:
    """Plot GA convergence: generation vs. best fitness score.
    
    Parameters
    ----------
    history : List[float]
        List of best fitness scores for each generation.
    output_path : Path
        Path to save the plot (PNG file).
    title : str, default="GA Convergence"
        Plot title.
    xlabel : str, default="Generation"
        X-axis label.
    ylabel : str, default="Best Fitness Score"
        Y-axis label.
    show_improvement : bool, default=True
        If True, highlight improvement points with markers.
    
    Returns
    -------
    Path
        Path to the saved plot file.
    """
    if not history:
        raise ValueError("History is empty, cannot plot convergence.")
    
    generations = np.arange(1, len(history) + 1)
    fitness_scores = np.array(history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main convergence curve
    ax.plot(generations, fitness_scores, 'b-', linewidth=2, label='Best Fitness', alpha=0.7)
    
    # Highlight improvement points
    if show_improvement and len(history) > 1:
        improvements = []
        best_so_far = fitness_scores[0]
        for i, score in enumerate(fitness_scores):
            if score > best_so_far:
                improvements.append(i)
                best_so_far = score
        
        if improvements:
            improvement_gens = [generations[i] for i in improvements]
            improvement_scores = [fitness_scores[i] for i in improvements]
            ax.scatter(
                improvement_gens,
                improvement_scores,
                c='red',
                s=50,
                marker='o',
                zorder=5,
                label='Improvement',
                edgecolors='darkred',
                linewidths=1.5,
            )
    
    # Add best point annotation
    best_idx = np.argmax(fitness_scores)
    best_gen = generations[best_idx]
    best_score = fitness_scores[best_idx]
    ax.scatter(
        [best_gen],
        [best_score],
        c='gold',
        s=150,
        marker='*',
        zorder=6,
        label=f'Best (Gen {best_gen}: {best_score:.4f})',
        edgecolors='orange',
        linewidths=2,
    )
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    
    # Add statistics text
    stats_text = (
        f"Initial: {fitness_scores[0]:.4f}\n"
        f"Final: {fitness_scores[-1]:.4f}\n"
        f"Best: {best_score:.4f}\n"
        f"Improvement: {((best_score - fitness_scores[0]) / abs(fitness_scores[0]) * 100):.2f}%"
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_solver_convergence(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Solver Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    method: Optional[str] = None,
) -> Path:
    """Plot solver convergence: iteration vs. loss.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing 'loss' key with list of loss values per iteration.
    output_path : Path
        Path to save the plot (PNG file).
    title : str, default="Solver Convergence"
        Plot title.
    xlabel : str, default="Iteration"
        X-axis label.
    ylabel : str, default="Loss"
        Y-axis label.
    method : str, optional
        Solver method name (e.g., 'adam', 'bfgs') to include in title.
    
    Returns
    -------
    Path
        Path to the saved plot file.
    """
    if "loss" not in history or not history["loss"]:
        raise ValueError("Loss history is empty, cannot plot convergence.")
    
    losses = np.array(history["loss"])
    iterations = np.arange(1, len(losses) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss curve
    ax.plot(iterations, losses, 'b-', linewidth=2, label='Loss', alpha=0.7)
    
    # Highlight convergence (if loss decreases significantly)
    if len(losses) > 1:
        # Find significant improvements
        improvements = []
        best_so_far = losses[0]
        for i, loss in enumerate(losses):
            if loss < best_so_far * 0.99:  # At least 1% improvement
                improvements.append(i)
                best_so_far = loss
        
        if improvements:
            improvement_iters = [iterations[i] for i in improvements[:10]]  # Limit to first 10
            improvement_losses = [losses[i] for i in improvements[:10]]
            ax.scatter(
                improvement_iters,
                improvement_losses,
                c='green',
                s=40,
                marker='o',
                zorder=5,
                label='Significant Improvement',
                alpha=0.7,
            )
    
    # Add best point annotation
    best_idx = np.argmin(losses)
    best_iter = iterations[best_idx]
    best_loss = losses[best_idx]
    ax.scatter(
        [best_iter],
        [best_loss],
        c='red',
        s=150,
        marker='*',
        zorder=6,
        label=f'Best (Iter {best_iter}: {best_loss:.6f})',
        edgecolors='darkred',
        linewidths=2,
    )
    
    # Use log scale if loss range is large
    if losses.max() / losses.min() > 100:
        ax.set_yscale('log')
        ylabel = f"{ylabel} (log scale)"
    
    method_str = f" ({method.upper()})" if method else ""
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(f"{title}{method_str}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    
    # Add statistics text
    reduction = ((losses[0] - best_loss) / losses[0] * 100) if losses[0] > 0 else 0
    stats_text = (
        f"Initial Loss: {losses[0]:.6f}\n"
        f"Final Loss: {losses[-1]:.6f}\n"
        f"Best Loss: {best_loss:.6f}\n"
        f"Reduction: {reduction:.2f}%\n"
        f"Converged at: Iter {best_iter}"
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
    )
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_enhanced_ga_convergence(
    fitness_history: List[float],
    diversity_history: Optional[List[float]],
    output_path: Path,
    title: str = "Enhanced GA Convergence",
) -> Path:
    """Plot enhanced GA convergence with fitness and diversity.
    
    Parameters
    ----------
    fitness_history : List[float]
        List of best fitness scores for each generation.
    diversity_history : List[float], optional
        List of population diversity values for each generation.
    output_path : Path
        Path to save the plot (PNG file).
    title : str, default="Enhanced GA Convergence"
        Plot title.
    
    Returns
    -------
    Path
        Path to the saved plot file.
    """
    if not fitness_history:
        raise ValueError("Fitness history is empty, cannot plot convergence.")
    
    generations = np.arange(1, len(fitness_history) + 1)
    fitness_scores = np.array(fitness_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Fitness convergence
    ax1.plot(generations, fitness_scores, 'b-', linewidth=2, label='Best Fitness', alpha=0.7)
    
    best_idx = np.argmax(fitness_scores)
    best_gen = generations[best_idx]
    best_score = fitness_scores[best_idx]
    ax1.scatter(
        [best_gen],
        [best_score],
        c='gold',
        s=150,
        marker='*',
        zorder=6,
        label=f'Best (Gen {best_gen}: {best_score:.4f})',
        edgecolors='orange',
        linewidths=2,
    )
    
    ax1.set_ylabel('Best Fitness Score', fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Diversity (if available)
    if diversity_history and len(diversity_history) == len(fitness_history):
        diversity_scores = np.array(diversity_history)
        ax2.plot(generations, diversity_scores, 'g-', linewidth=2, label='Population Diversity', alpha=0.7)
        ax2.set_ylabel('Diversity (Normalized Hamming Distance)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9)
    else:
        ax2.axis('off')
        ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
    
    # Add statistics
    stats_text = (
        f"Initial: {fitness_scores[0]:.4f}\n"
        f"Final: {fitness_scores[-1]:.4f}\n"
        f"Best: {best_score:.4f}\n"
        f"Improvement: {((best_score - fitness_scores[0]) / abs(fitness_scores[0]) * 100):.2f}%"
    )
    ax1.text(
        0.02, 0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_combined_convergence(
    ga_history: Optional[List[float]],
    solver_history: Optional[Dict[str, List[float]]],
    output_path: Path,
    ga_title: str = "GA Convergence",
    solver_title: str = "Solver Convergence",
    solver_method: Optional[str] = None,
) -> Path:
    """Plot both GA and Solver convergence in a single figure.
    
    Parameters
    ----------
    ga_history : List[float], optional
        GA fitness history (generations).
    solver_history : Dict[str, List[float]], optional
        Solver loss history (iterations).
    output_path : Path
        Path to save the combined plot (PNG file).
    ga_title : str, default="GA Convergence"
        Title for GA subplot.
    solver_title : str, default="Solver Convergence"
        Title for Solver subplot.
    solver_method : str, optional
        Solver method name.
    
    Returns
    -------
    Path
        Path to the saved plot file.
    """
    n_plots = sum([ga_history is not None, solver_history is not None])
    if n_plots == 0:
        raise ValueError("At least one history (GA or Solver) must be provided.")
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 6 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot GA convergence
    if ga_history:
        generations = np.arange(1, len(ga_history) + 1)
        fitness_scores = np.array(ga_history)
        
        axes[plot_idx].plot(generations, fitness_scores, 'b-', linewidth=2, label='Best Fitness', alpha=0.7)
        
        best_idx = np.argmax(fitness_scores)
        best_gen = generations[best_idx]
        best_score = fitness_scores[best_idx]
        axes[plot_idx].scatter(
            [best_gen],
            [best_score],
            c='gold',
            s=150,
            marker='*',
            zorder=6,
            label=f'Best (Gen {best_gen}: {best_score:.4f})',
            edgecolors='orange',
            linewidths=2,
        )
        
        axes[plot_idx].set_xlabel('Generation', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Best Fitness Score', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(ga_title, fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
        axes[plot_idx].legend(loc='best', fontsize=9)
        plot_idx += 1
    
    # Plot Solver convergence
    if solver_history and "loss" in solver_history:
        losses = np.array(solver_history["loss"])
        iterations = np.arange(1, len(losses) + 1)
        
        axes[plot_idx].plot(iterations, losses, 'r-', linewidth=2, label='Loss', alpha=0.7)
        
        best_idx = np.argmin(losses)
        best_iter = iterations[best_idx]
        best_loss = losses[best_idx]
        axes[plot_idx].scatter(
            [best_iter],
            [best_loss],
            c='red',
            s=150,
            marker='*',
            zorder=6,
            label=f'Best (Iter {best_iter}: {best_loss:.6f})',
            edgecolors='darkred',
            linewidths=2,
        )
        
        if losses.max() / losses.min() > 100:
            axes[plot_idx].set_yscale('log')
        
        method_str = f" ({solver_method.upper()})" if solver_method else ""
        axes[plot_idx].set_xlabel('Iteration', fontsize=11, fontweight='bold')
        axes[plot_idx].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[plot_idx].set_title(f"{solver_title}{method_str}", fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
        axes[plot_idx].legend(loc='best', fontsize=9)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

