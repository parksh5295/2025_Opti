"""Experiment tracking utilities for pipeline logging and resumption."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump, load


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StageData:
    name: str
    artifacts: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


class ExperimentTracker:
    """Manage experiment directories, logging, and stage persistence."""

    def __init__(
        self,
        data_path: Path,
        method_label: str,
        run_name: Optional[str] = None,
        resume_state: Optional[Path] = None,
    ) -> None:
        self.data_path = Path(data_path).resolve()
        if resume_state is not None:
            resume_state = resume_state.resolve()

        self.method_label = method_label
        self.data_root = self.compute_data_root(self.data_path)
        self.results_root = self.data_root / "Result" / self.method_label
        self.logs_root = self.data_root / "log" / self.method_label
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

        if resume_state is not None:
            state_path = resume_state
            if state_path.is_dir():
                state_path = state_path / "state.json"
            if not state_path.exists():
                raise FileNotFoundError(f"State file not found at {state_path}")
            with state_path.open("r", encoding="utf-8") as fh:
                self.state = json.load(fh)

            self.run_name = self.state["run_name"]
            self.result_dir = Path(self.state["result_dir"]).resolve()
            self.log_dir = Path(self.state["log_dir"]).resolve()
            self.state_path = self.log_dir / "state.json"
            if self.state.get("method_label") != self.method_label:
                raise ValueError(
                    "Method label mismatch between resume state and current configuration."
                )
            self.state["status"] = "running"
            self.state.setdefault("resumed_at", _timestamp())
            self._write_state()
        else:
            self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.result_dir = self.results_root / self.run_name
            self.log_dir = self.logs_root / self.run_name
            self.result_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.state_path = self.log_dir / "state.json"
            self.state = {
                "run_name": self.run_name,
                "method_label": self.method_label,
                "created_at": _timestamp(),
                "result_dir": str(self.result_dir.resolve()),
                "log_dir": str(self.log_dir.resolve()),
                "status": "running",
                "stages": {},
            }
            self._write_state()

        self.logger = self._configure_logger()
        self.events_path = self.log_dir / "events.csv"
        if not self.events_path.exists():
            with self.events_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["timestamp", "stage", "message", "extra"])
                writer.writeheader()

    @classmethod
    def compute_data_root(cls, data_path: Path) -> Path:
        data_path = Path(data_path).resolve()
        parents = list(data_path.parents)
        if len(parents) >= 2:
            return parents[1]
        return data_path.parent

    @classmethod
    def find_latest_state(cls, data_path: Path, method_label: str) -> Optional[Path]:
        data_root = cls.compute_data_root(data_path)
        log_root = data_root / "log" / method_label
        if not log_root.exists():
            return None

        candidates: list[tuple[datetime, Path]] = []
        for run_dir in log_root.iterdir():
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            status = state.get("status")
            if status in {"running", "interrupted", "failed"}:
                updated = state.get("updated_at") or state.get("created_at")
                try:
                    timestamp = datetime.fromisoformat(updated)
                except Exception:
                    timestamp = datetime.min
                candidates.append((timestamp, state_path))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"pipeline.{self.run_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            file_handler = logging.FileHandler(self.log_dir / "run.log", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(stream_handler)

        return logger

    def mark_status(self, status: str) -> None:
        self.state["status"] = status
        self.state["updated_at"] = _timestamp()
        self._write_state()

    def log_event(self, stage: str, message: str, extra: Optional[Dict[str, Any]] = None, level: int = logging.INFO) -> None:
        payload = {"timestamp": _timestamp(), "stage": stage, "message": message}
        if extra:
            payload["extra"] = json.dumps(extra, ensure_ascii=False)
        else:
            payload["extra"] = ""

        with self.events_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "stage", "message", "extra"])
            writer.writerow(payload)

        if extra:
            self.logger.log(level, f"[{stage}] {message} | extra={extra}")
        else:
            self.logger.log(level, f"[{stage}] {message}")

    def is_completed(self, stage: str) -> bool:
        return stage in self.state["stages"]

    def _write_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Stage persistence helpers
    # ------------------------------------------------------------------
    def save_preprocessing(self, data: Dict[str, Any]) -> None:
        stage = "preprocessing"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        for key, value in data.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                path = stage_dir / f"{key}.pkl"
                value.to_pickle(path)
                artifacts[key] = {"type": "pandas_pickle", "path": self._relpath(path)}
            elif key == "feature_columns":
                path = stage_dir / f"{key}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                artifacts[key] = {"type": "json", "path": self._relpath(path)}
            elif key == "scaler":
                path = stage_dir / "scaler.joblib"
                dump(value, path)
                artifacts[key] = {"type": "joblib", "path": self._relpath(path)}
            else:
                path = stage_dir / f"{key}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                artifacts[key] = {"type": "json", "path": self._relpath(path)}

        self._record_stage(stage, artifacts, {})

    def load_preprocessing(self) -> Dict[str, Any]:
        info = self.state["stages"]["preprocessing"]["artifacts"]
        stage_data: Dict[str, Any] = {}
        for key, meta in info.items():
            path = self.result_dir / meta["path"]
            if meta["type"] == "pandas_pickle":
                stage_data[key] = pd.read_pickle(path)
            elif meta["type"] == "json":
                stage_data[key] = json.loads(path.read_text(encoding="utf-8"))
            elif meta["type"] == "joblib":
                stage_data[key] = load(path)
        return stage_data

    def save_feature_scores(
        self,
        pca_scores: Dict[str, float],
        mi_scores: Dict[str, float],
        rf_scores: Dict[str, float],
        ensemble_scores: Dict[str, float],
    ) -> None:
        stage = "feature_scoring"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        for name, scores in {
            "pca_scores": pca_scores,
            "mi_scores": mi_scores,
            "rf_scores": rf_scores,
            "ensemble_scores": ensemble_scores,
        }.items():
            df = pd.DataFrame(scores.items(), columns=["feature", "score"])
            path = stage_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            artifacts[name] = {"type": "csv", "path": self._relpath(path)}

        self._record_stage(stage, artifacts, {})

    def load_feature_scores(self) -> Dict[str, Dict[str, float]]:
        info = self.state["stages"]["feature_scoring"]["artifacts"]
        output: Dict[str, Dict[str, float]] = {}
        for name, meta in info.items():
            path = self.result_dir / meta["path"]
            df = pd.read_csv(path)
            output[name.replace("_scores", "")] = dict(zip(df["feature"], df["score"]))
        return output

    def save_redundancy(
        self,
        penalty_matrix: pd.DataFrame,
        candidate_features: Dict[str, Any],
    ) -> None:
        stage = "redundancy"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        penalty_path = stage_dir / "penalty_matrix.pkl"
        penalty_matrix.to_pickle(penalty_path)
        artifacts["penalty_matrix"] = {"type": "pandas_pickle", "path": self._relpath(penalty_path)}

        candidate_path = stage_dir / "candidate_features.json"
        candidate_path.write_text(json.dumps(candidate_features), encoding="utf-8")
        artifacts["candidate_features"] = {"type": "json", "path": self._relpath(candidate_path)}

        self._record_stage(stage, artifacts, {"num_candidates": len(candidate_features)})

    def load_redundancy(self) -> Dict[str, Any]:
        info = self.state["stages"]["redundancy"]["artifacts"]
        penalty_matrix = pd.read_pickle(self.result_dir / info["penalty_matrix"]["path"])
        candidate_features = json.loads((self.result_dir / info["candidate_features"]["path"]).read_text(encoding="utf-8"))
        return {
            "penalty_matrix": penalty_matrix,
            "candidate_features": candidate_features,
        }

    def save_ga_results(
        self,
        selected_features: Dict[str, Any],
        best_score: float,
        history: Optional[list] = None,
        redundancy_penalty: Optional[float] = None,
    ) -> None:
        stage = "ga"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        features_path = stage_dir / "selected_features.json"
        features_path.write_text(json.dumps(selected_features), encoding="utf-8")
        artifacts["selected_features"] = {"type": "json", "path": self._relpath(features_path)}

        meta = {"best_score": best_score}
        if redundancy_penalty is not None:
            meta["redundancy_penalty"] = redundancy_penalty
        if history is not None:
            history_path = stage_dir / "ga_history.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            artifacts["ga_history"] = {"type": "json", "path": self._relpath(history_path)}

        self._record_stage(stage, artifacts, meta)

    def load_ga_results(self) -> Dict[str, Any]:
        stage_info = self.state["stages"]["ga"]
        artifacts = stage_info["artifacts"]
        selected_features = json.loads((self.result_dir / artifacts["selected_features"]["path"]).read_text(encoding="utf-8"))
        history = None
        if "ga_history" in artifacts:
            history = json.loads((self.result_dir / artifacts["ga_history"]["path"]).read_text(encoding="utf-8"))
        meta = stage_info.get("metadata", {})
        return {
            "selected_features": selected_features,
            "best_score": meta.get("best_score"),
            "redundancy_penalty": meta.get("redundancy_penalty"),
            "history": history,
        }

    def save_solver_results(
        self,
        backend: str,
        weights: Optional[np.ndarray],
        bias: Optional[float],
        threshold: float,
        val_score: float,
        solver_details: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> None:
        stage = "solver"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        if backend == "custom" and weights is not None and bias is not None:
            weights_path = stage_dir / "weights.npy"
            np.save(weights_path, weights)
            artifacts["weights"] = {"type": "npy", "path": self._relpath(weights_path)}

            bias_path = stage_dir / "bias.json"
            bias_path.write_text(json.dumps({"bias": bias}), encoding="utf-8")
            artifacts["bias"] = {"type": "json", "path": self._relpath(bias_path)}
        elif backend == "sklearn" and model is not None:
            model_path = stage_dir / "model.joblib"
            dump(model, model_path)
            artifacts["model"] = {"type": "joblib", "path": self._relpath(model_path)}

        threshold_path = stage_dir / "threshold.json"
        threshold_path.write_text(json.dumps({"threshold": threshold, "val_score": val_score}), encoding="utf-8")
        artifacts["threshold"] = {"type": "json", "path": self._relpath(threshold_path)}

        details_path = stage_dir / "solver_details.json"
        details_path.write_text(json.dumps({"backend": backend, **solver_details}), encoding="utf-8")
        artifacts["solver_details"] = {"type": "json", "path": self._relpath(details_path)}

        metadata = {"backend": backend, "val_score": val_score}
        self._record_stage(stage, artifacts, metadata)

    def load_solver_results(self) -> Dict[str, Any]:
        stage_info = self.state["stages"]["solver"]
        artifacts = stage_info["artifacts"]
        backend = stage_info["metadata"]["backend"]
        threshold_meta = json.loads((self.result_dir / artifacts["threshold"]["path"]).read_text(encoding="utf-8"))

        results: Dict[str, Any] = {
            "backend": backend,
            "threshold": threshold_meta["threshold"],
            "val_score": threshold_meta["val_score"],
            "solver_details": json.loads((self.result_dir / artifacts["solver_details"]["path"]).read_text(encoding="utf-8")),
        }

        if backend == "custom":
            weights = np.load(self.result_dir / artifacts["weights"]["path"])
            bias_meta = json.loads((self.result_dir / artifacts["bias"]["path"]).read_text(encoding="utf-8"))
            results["weights"] = weights
            results["bias"] = bias_meta["bias"]
        elif backend == "sklearn":
            results["model"] = load(self.result_dir / artifacts["model"]["path"])

        return results

    def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        stage = "evaluation"
        stage_dir = self.result_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Dict[str, Any]] = {}

        eval_path = stage_dir / "evaluation.json"
        eval_path.write_text(json.dumps(evaluation, default=self._json_default, indent=2), encoding="utf-8")
        artifacts["evaluation"] = {"type": "json", "path": self._relpath(eval_path)}

        metrics_df = pd.DataFrame(
            [
                {"metric": "roc_auc", "value": evaluation["roc_auc"]},
                {"metric": "pr_auc", "value": evaluation["pr_auc"]},
                {"metric": "f1", "value": evaluation["f1"]},
                {"metric": "overall_score", "value": evaluation["overall_score"]},
            ]
        )
        metrics_path = stage_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        artifacts["metrics"] = {"type": "csv", "path": self._relpath(metrics_path)}

        report_path = stage_dir / "classification_report.txt"
        report_path.write_text(evaluation["classification_report"], encoding="utf-8")
        artifacts["classification_report"] = {"type": "text", "path": self._relpath(report_path)}

        matrix_path = stage_dir / "confusion_matrix.csv"
        pd.DataFrame(evaluation["confusion_matrix"]).to_csv(matrix_path, index=False)
        artifacts["confusion_matrix"] = {"type": "csv", "path": self._relpath(matrix_path)}

        self._record_stage(stage, artifacts, {})

    def load_evaluation(self) -> Dict[str, Any]:
        info = self.state["stages"]["evaluation"]["artifacts"]
        evaluation = json.loads((self.result_dir / info["evaluation"]["path"]).read_text(encoding="utf-8"))
        return evaluation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_stage(self, stage: str, artifacts: Dict[str, Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        self.state["stages"][stage] = {
            "completed_at": _timestamp(),
            "artifacts": artifacts,
            "metadata": metadata,
        }
        self._write_state()

    def _relpath(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.result_dir.resolve()))

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


