"""
Experiment tracking for R-JEPA using MLflow.

Provides a lightweight ``ExperimentTracker`` wrapper that can be
enabled / disabled at runtime, making it safe to use even when MLflow
is not installed in the environment (the tracker degrades to a no-op).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed — experiment tracking disabled.")


class ExperimentTracker:
    """
    Thin wrapper around MLflow for experiment tracking.

    All methods are safe to call regardless of whether MLflow is available
    or the tracker is enabled — calls degrade to no-ops when disabled.

    Usage::

        tracker = ExperimentTracker(
            experiment_name="r-jepa",
            tracking_uri="http://localhost:5000",
            tags={"model": "r_jepa", "dataset": "daily"},
        )
        tracker.log_params({"lr": 0.001, "batch_size": 64})
        tracker.log_metrics({"loss": 0.05, "val_loss": 0.06}, step=1)
        tracker.log_artifact("/path/to/checkpoint_best.pt")
        tracker.end_run()
    """

    def __init__(
        self,
        enabled: bool = True,
        experiment_name: str = "r-jepa",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
        artifact_location: str | None = None,
    ) -> None:
        """
        Args:
            enabled: Set ``False`` to disable all tracking (no-op mode).
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI (``None`` = local
                ``mlruns/`` directory).
            run_name: Human-readable name for this run (auto-generated
                if ``None``).
            tags: Key-value tags attached to the run.
            artifact_location: Local or S3 path where artifacts are stored
                (``None`` = MLflow default).
        """
        self.enabled = enabled and _MLFLOW_AVAILABLE
        self._active_run = False

        if not self.enabled:
            return

        # Set tracking URI (local ./mlruns by default)
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location,
                )
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
        except Exception as exc:
            logger.warning("Failed to initialise MLflow experiment: %s", exc)
            self.enabled = False
            return

        # Start a new run
        try:
            mlflow.start_run(run_name=run_name)
            self._active_run = True
            if tags:
                mlflow.set_tags(dict(tags))
        except Exception as exc:
            logger.warning("Failed to start MLflow run: %s", exc)
            self.enabled = False

    # ── Public API ─────────────────────────────────────────────────

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log a dictionary of hyper-parameters."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.log_params(self._flatten(params))
        except Exception as exc:
            logger.warning("Failed to log params: %s", exc)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log a dictionary of scalar metrics at an optional step."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.log_metrics(dict(metrics), step=step)
        except Exception as exc:
            logger.warning("Failed to log metrics: %s", exc)

    def log_artifact(self, local_path: str) -> None:
        """Upload a single file as an MLflow artifact."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.log_artifact(local_path)
        except Exception as exc:
            logger.warning("Failed to log artifact: %s", exc)

    def log_artifacts(self, local_dir: str) -> None:
        """Upload an entire directory as MLflow artifacts."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.log_artifacts(local_dir)
        except Exception as exc:
            logger.warning("Failed to log artifacts: %s", exc)

    def log_dict(self, dictionary: dict[str, Any], artifact_path: str) -> None:
        """Log a JSON-serialisable dictionary as an artifact file."""
        if not self.enabled or not self._active_run:
            return
        tmp_path: str = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(dictionary, f, indent=2, default=str)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("Failed to log dict as artifact: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on the current run."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.set_tag(key, value)
        except Exception as exc:
            logger.warning("Failed to set tag: %s", exc)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.end_run()
        except Exception as exc:
            logger.warning("Failed to end MLflow run: %s", exc)
        self._active_run = False

    def __enter__(self) -> ExperimentTracker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_run()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _flatten(
        d: Mapping[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, str]:
        """Flatten nested dict into dot-separated keys for MLflow params."""
        items: list[tuple[str, str]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, Mapping):
                items.extend(ExperimentTracker._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)
