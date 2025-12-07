"""Transfer learning utilities for GA initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from Module.experiment_tracker import ExperimentTracker


def load_previous_best_solutions(
    data_path: Path,
    method_label: str,
    candidate_features: List[str],
) -> Optional[List[np.ndarray]]:
    """Load previous best solutions for transfer learning.
    
    Parameters
    ----------
    data_path:
        Path to the dataset.
    method_label:
        Method label (e.g., 'with_hessian', 'without_hessian').
    candidate_features:
        List of candidate feature names.
    
    Returns
    -------
    Optional list of binary feature vectors (numpy arrays), or None if not found.
    """
    try:
        base_root = ExperimentTracker.compute_data_root(data_path)
        log_root = base_root / "log" / method_label
        
        if not log_root.exists():
            return None
        
        # Find latest completed run
        latest_state = ExperimentTracker.find_latest_completed_state(data_path, method_label)
        if latest_state is None:
            return None
        
        result_dir = Path(
            latest_state.parent.parent.parent / "Result" / method_label / latest_state.parent.name
        )
        ga_dir = result_dir / "ga"
        
        if not ga_dir.exists():
            return None
        
        # Load selected features
        selected_features_path = ga_dir / "selected_features.json"
        if not selected_features_path.exists():
            return None
        
        selected_features = json.loads(selected_features_path.read_text(encoding="utf-8"))
        
        # Convert to binary representation
        feature_to_idx = {feat: idx for idx, feat in enumerate(candidate_features)}
        solution = np.zeros(len(candidate_features), dtype=int)
        for feat in selected_features:
            if feat in feature_to_idx:
                solution[feature_to_idx[feat]] = 1
        
        return [solution] if solution.sum() > 0 else None
    except Exception:
        return None

