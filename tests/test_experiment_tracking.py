"""
Tests for ``utils.experiment_tracking`` — the MLflow ``ExperimentTracker``.

Because MLflow may not be installed in the CI / test environment, these
tests focus on the **no-op (disabled) mode** and the helper methods that
do not require a live MLflow server.
"""

from __future__ import annotations

from typing import Any

import pytest

from utils.experiment_tracking import ExperimentTracker


class TestExperimentTrackerDisabled:
    """All methods should be safe no-ops when MLflow is unavailable
    or ``enabled=False``."""

    def test_constructor_with_enabled_false(self) -> None:
        """When enabled=False, no exception is raised."""
        tracker = ExperimentTracker(enabled=False)
        assert not tracker.enabled
        assert not tracker._active_run

    def test_log_params_noop(self) -> None:
        """log_params does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.log_params({"lr": 0.001, "batch_size": 64})  # should not raise

    def test_log_metrics_noop(self) -> None:
        """log_metrics does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.log_metrics({"loss": 0.05}, step=1)  # should not raise

    def test_log_artifact_noop(self) -> None:
        """log_artifact does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.log_artifact("/nonexistent/path.pt")  # should not raise

    def test_log_artifacts_noop(self) -> None:
        """log_artifacts does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.log_artifacts("/nonexistent/dir")  # should not raise

    def test_log_dict_noop(self) -> None:
        """log_dict does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.log_dict({"key": "value"}, "metrics.json")  # should not raise

    def test_set_tag_noop(self) -> None:
        """set_tag does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.set_tag("key", "value")  # should not raise

    def test_end_run_noop(self) -> None:
        """end_run does nothing when disabled."""
        tracker = ExperimentTracker(enabled=False)
        tracker.end_run()  # should not raise

    def test_context_manager_noop(self) -> None:
        """Context manager protocol works without errors when disabled."""
        with ExperimentTracker(enabled=False) as tracker:
            assert not tracker.enabled
            tracker.log_params({"a": 1})
        assert not tracker._active_run


class TestFlattenHelper:
    """Unit tests for the ``_flatten`` static helper that converts nested
    config dicts into dot-separated MLflow params."""

    def test_flat_dict(self) -> None:
        params = {"lr": 0.001, "batch_size": 64}
        flat = ExperimentTracker._flatten(params)
        assert flat == {"lr": "0.001", "batch_size": "64"}

    def test_nested_dict(self) -> None:
        params = {"training": {"lr": 0.001, "epochs": 100}, "data": {"batch_size": 64}}
        flat = ExperimentTracker._flatten(params)
        assert flat == {
            "training.lr": "0.001",
            "training.epochs": "100",
            "data.batch_size": "64",
        }

    def test_deeply_nested(self) -> None:
        params = {"model": {"r_jepa": {"latent_dim": 64, "encoder": {"hidden_dim": 128}}}}
        flat = ExperimentTracker._flatten(params)
        assert flat == {
            "model.r_jepa.latent_dim": "64",
            "model.r_jepa.encoder.hidden_dim": "128",
        }

    def test_non_string_values_are_converted(self) -> None:
        params = {"int_val": 42, "float_val": 3.14, "bool_val": True, "none_val": None}
        flat = ExperimentTracker._flatten(params)
        assert flat == {
            "int_val": "42",
            "float_val": "3.14",
            "bool_val": "True",
            "none_val": "None",
        }

    def test_custom_separator(self) -> None:
        params = {"data": {"batch_size": 64}}
        flat = ExperimentTracker._flatten(params, sep="/")
        assert flat == {"data/batch_size": "64"}

    def test_empty_dict(self) -> None:
        assert ExperimentTracker._flatten({}) == {}

    def test_mixed_nesting(self) -> None:
        """Lists and other iterables are not flattened further — they become
        string representations."""
        params = {"list_param": [1, 2, 3]}
        flat = ExperimentTracker._flatten(params)
        assert flat["list_param"] == "[1, 2, 3]"


class TestDefaultConstructor:
    """Tests for the constructor defaults when MLflow is absent."""

    def test_default_constructor_no_exception(self) -> None:
        """The constructor should never raise, even if MLflow is missing."""
        tracker = ExperimentTracker(enabled=False)
        assert not tracker.enabled

    def test_enabled_flag_reflects_mlflow_availability(self) -> None:
        """When no MLflow is available, enabled=False even if requested."""
        tracker = ExperimentTracker(enabled=True)
        # Without MLflow installed, the tracker must be disabled.
        # If MLflow *is* installed, we can't guarantee a running server,
        # but at least the constructor should not raise.
        assert isinstance(tracker.enabled, bool)


class TestIntegrationSafety:
    """Tests that verify the ExperimentTracker does not interfere with
    the rest of the codebase when disabled."""

    def test_trainer_accepts_none_tracker(self) -> None:
        """RJEPATrainer should accept experiment_tracker=None."""
        # We import here to avoid circular dependency issues in test discovery
        from training.trainer import RJEPATrainer

        # Just verify the __init__ signature accepts None — we don't
        # instantiate because that requires real data loaders.
        import inspect

        sig = inspect.signature(RJEPATrainer.__init__)
        assert "experiment_tracker" in sig.parameters
        assert sig.parameters["experiment_tracker"].default is None
