"""
Tests for data pipeline components.

Covers:
- ``StockDataset`` — context-target pair generation, augmenter integration
- ``data.augmentation`` — GaussianNoise, MagnitudeWarping, TimeWarping,
  WindowSlice, Compose
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from data.augmentation import (
    Compose,
    GaussianNoise,
    MagnitudeWarping,
    TimeWarping,
    WindowSlice,
)
from data.stock_data import StockDataConfig, StockDataset, create_dataloaders


# ── StockDataset ─────────────────────────────────────────────────────────


class TestStockDataset:
    """Tests for the time series dataset class."""

    def test_dataset_length(self, synthetic_data: np.ndarray) -> None:
        """Dataset length is correct given stride=1."""
        seq_len, pred_hz = 20, 5
        n = len(synthetic_data)
        expected = n - seq_len - pred_hz + 1
        dataset = StockDataset(data=synthetic_data, sequence_length=seq_len,
                               prediction_horizon=pred_hz)
        assert len(dataset) == expected, (
            f"Expected {expected} samples, got {len(dataset)}"
        )

    def test_context_target_shapes(self, synthetic_data: np.ndarray) -> None:
        """Context and target have correct shapes."""
        seq_len, pred_hz, n_features = 20, 5, synthetic_data.shape[1]
        dataset = StockDataset(data=synthetic_data, sequence_length=seq_len,
                               prediction_horizon=pred_hz)
        context, target = dataset[0]
        assert context.shape == (seq_len, n_features), (
            f"Context shape mismatch: {context.shape}"
        )
        assert target.shape == (pred_hz, n_features), (
            f"Target shape mismatch: {target.shape}"
        )

    def test_context_target_non_overlapping(self, synthetic_data: np.ndarray) -> None:
        """Context and target are non-overlapping (contiguous slices)."""
        seq_len, pred_hz = 20, 5
        dataset = StockDataset(data=synthetic_data, sequence_length=seq_len,
                               prediction_horizon=pred_hz)
        context, target = dataset[50]  # arbitrary index
        # Target starts right after context ends
        assert torch.equal(
            target,
            torch.from_numpy(
                synthetic_data[50 + seq_len : 50 + seq_len + pred_hz]
            ).float(),
        ), "Target should be the slice immediately following context"

    def test_stride_affects_length(self, synthetic_data: np.ndarray) -> None:
        """Larger stride produces fewer samples."""
        seq_len, pred_hz = 20, 5
        ds1 = StockDataset(data=synthetic_data, sequence_length=seq_len,
                           prediction_horizon=pred_hz, stride=1)
        ds2 = StockDataset(data=synthetic_data, sequence_length=seq_len,
                           prediction_horizon=pred_hz, stride=3)
        assert len(ds2) < len(ds1), "Higher stride should reduce dataset length"

    def test_augmenter_applied_to_context_only(self, synthetic_data: np.ndarray) -> None:
        """Augmenter modifies context but not target."""
        seq_len, pred_hz = 20, 5
        noise = GaussianNoise(std=0.5)
        dataset = StockDataset(data=synthetic_data, sequence_length=seq_len,
                               prediction_horizon=pred_hz, augmenter=noise)
        context, target = dataset[0]
        # With high noise, context should differ from original
        original_context = torch.from_numpy(synthetic_data[0:seq_len]).float()
        assert not torch.allclose(context, original_context, atol=0.3), (
            "Augmented context should differ from original"
        )
        # Target should be unchanged
        original_target = torch.from_numpy(
            synthetic_data[seq_len:seq_len + pred_hz]
        ).float()
        assert torch.equal(target, original_target), (
            "Target should be identical to original (no augmentation)"
        )

    def test_no_augmenter_preserves_data(self, synthetic_data: np.ndarray) -> None:
        """Without augmenter, context and target match original data exactly."""
        seq_len, pred_hz = 20, 5
        dataset = StockDataset(data=synthetic_data, sequence_length=seq_len,
                               prediction_horizon=pred_hz, augmenter=None)
        context, target = dataset[10]
        expected_context = torch.from_numpy(
            synthetic_data[10:10 + seq_len]
        ).float()
        expected_target = torch.from_numpy(
            synthetic_data[10 + seq_len:10 + seq_len + pred_hz]
        ).float()
        assert torch.equal(context, expected_context)
        assert torch.equal(target, expected_target)

    def test_dataset_iterable(self, synthetic_data: np.ndarray) -> None:
        """Dataset can be iterated over with a DataLoader."""
        from torch.utils.data import DataLoader
        dataset = StockDataset(data=synthetic_data, sequence_length=20,
                               prediction_horizon=5)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batches = list(loader)
        assert len(batches) > 0, "DataLoader should yield at least one batch"
        context_batch, target_batch = batches[0]
        assert context_batch.ndim == 3  # [B, seq_len, n_features]
        assert target_batch.ndim == 3   # [B, pred_hz, n_features]


# ── GaussianNoise ────────────────────────────────────────────────────────


class TestGaussianNoise:
    """Tests for the GaussianNoise augmentation."""

    def test_output_shape(self) -> None:
        """Output has the same shape as input."""
        x = torch.randn(4, 16, 5)
        transform = GaussianNoise(std=0.1)
        out = transform(x)
        assert out.shape == x.shape

    def test_noise_added(self) -> None:
        """Output differs from input."""
        x = torch.randn(4, 16, 5)
        transform = GaussianNoise(std=0.5)
        out = transform(x)
        assert not torch.equal(out, x), "Noise should change the tensor"

    def test_zero_std_no_change(self) -> None:
        """std=0.0 leaves input unchanged."""
        x = torch.randn(4, 16, 5)
        transform = GaussianNoise(std=0.0)
        out = transform(x)
        assert torch.equal(out, x), "std=0 should preserve input"

    def test_repr(self) -> None:
        transform = GaussianNoise(std=0.02)
        r = repr(transform)
        assert "GaussianNoise" in r
        assert "0.02" in r


# ── MagnitudeWarping ─────────────────────────────────────────────────────


class TestMagnitudeWarping:
    """Tests for the MagnitudeWarping augmentation."""

    def test_output_shape(self) -> None:
        x = torch.ones(10, 3)  # [seq_len, n_features]
        transform = MagnitudeWarping(sigma=0.2, knot_count=4)
        out = transform(x)
        assert out.shape == x.shape

    def test_warping_changes_values(self) -> None:
        x = torch.ones(10, 3)  # [seq_len, n_features]
        transform = MagnitudeWarping(sigma=0.5, knot_count=4)
        out = transform(x)
        # With sigma=0.5 and all-ones input, output should differ from 1.0
        assert not torch.allclose(out, x, atol=0.05)

    def test_zero_sigma_no_change(self) -> None:
        x = torch.randn(10, 3)  # [seq_len, n_features]
        transform = MagnitudeWarping(sigma=0.0, knot_count=5)
        out = transform(x)
        assert torch.equal(out, x), "sigma=0 should preserve input"

    def test_repr(self) -> None:
        transform = MagnitudeWarping(sigma=0.1)
        r = repr(transform)
        assert "MagnitudeWarping" in r


# ── TimeWarping ──────────────────────────────────────────────────────────


class TestTimeWarping:
    """Tests for the TimeWarping augmentation."""

    def test_output_shape(self) -> None:
        x = torch.randn(10, 3)  # [seq_len, n_features]
        transform = TimeWarping(sigma=0.2)
        out = transform(x)
        assert out.shape == x.shape

    def test_zero_sigma_no_change(self) -> None:
        x = torch.randn(10, 3)  # [seq_len, n_features]
        transform = TimeWarping(sigma=0.0)
        out = transform(x)
        assert torch.equal(out, x), "sigma=0 should preserve input"

    def test_repr(self) -> None:
        transform = TimeWarping(sigma=0.15)
        r = repr(transform)
        assert "TimeWarping" in r


# ── WindowSlice ──────────────────────────────────────────────────────────


class TestWindowSlice:
    """Tests for the WindowSlice augmentation."""

    def test_output_shape(self) -> None:
        x = torch.randn(20, 5)  # [seq_len, n_features]
        transform = WindowSlice(ratio=0.3)
        out = transform(x)
        assert out.shape == x.shape, "Output shape should match input"

    def test_zero_ratio_no_change(self) -> None:
        x = torch.randn(20, 5)  # [seq_len, n_features]
        transform = WindowSlice(ratio=0.0)
        out = transform(x)
        assert torch.equal(out, x), "ratio=0 should preserve input"

    def test_repr(self) -> None:
        transform = WindowSlice(ratio=0.15)
        r = repr(transform)
        assert "WindowSlice" in r


# ── Compose ──────────────────────────────────────────────────────────────


class TestCompose:
    """Tests for the Compose meta-transform."""

    def test_compose_applies_all_transforms(self) -> None:
        x = torch.ones(10, 3)  # [seq_len, n_features]
        transform = Compose([
            GaussianNoise(std=0.1),
            MagnitudeWarping(sigma=0.1, knot_count=4),
        ])
        out = transform(x)
        assert out.shape == x.shape
        # At least some change should have occurred
        assert not torch.equal(out, x)

    def test_empty_compose_preserves_input(self) -> None:
        x = torch.randn(10, 3)  # [seq_len, n_features]
        transform = Compose([])
        out = transform(x)
        assert torch.equal(out, x)

    def test_repr(self) -> None:
        transform = Compose([GaussianNoise(std=0.02)])
        r = repr(transform)
        assert "Compose" in r
        assert "GaussianNoise" in r


# ── create_dataloaders ───────────────────────────────────────────────────


class TestCreateDataloaders:
    """Tests for the dataloader factory function."""

    @pytest.fixture
    def config(self) -> StockDataConfig:
        return StockDataConfig(
            tickers=("TEST",),
            start_date="2020-01-01",
            end_date="2020-12-31",
            sequence_length=20,
            prediction_horizon=5,
            features=("Open", "High", "Low", "Close", "Volume"),
            normalize=False,
            train_split=0.8,
            val_split=0.1,
        )

    def test_returns_three_loaders(self, synthetic_data: np.ndarray,
                                    config: StockDataConfig) -> None:
        train, val, test = create_dataloaders(synthetic_data, config,
                                              batch_size=4, num_workers=0)
        assert train is not None
        assert val is not None
        assert test is not None

    def test_train_val_test_partition(self, synthetic_data: np.ndarray,
                                       config: StockDataConfig) -> None:
        train, val, test = create_dataloaders(synthetic_data, config,
                                              batch_size=4, num_workers=0)
        # Check that they have different sizes
        n_train = len(train.dataset)  # type: ignore[arg-type]
        n_val = len(val.dataset)  # type: ignore[arg-type]
        n_test = len(test.dataset)  # type: ignore[arg-type]
        assert n_train > 0
        assert n_val > 0
        assert n_test > 0
        # Train should be the largest partition
        assert n_train > n_val
        assert n_train > n_test

    def test_no_augmentation_on_val_test(self, synthetic_data: np.ndarray,
                                          config: StockDataConfig) -> None:
        """Validation and test datasets should have augmenter=None."""
        train, val, test = create_dataloaders(synthetic_data, config,
                                              batch_size=4, num_workers=0)
        assert train.dataset.augmenter is not None or config.aug_noise_std == 0.0
        assert val.dataset.augmenter is None  # type: ignore[union-attr]
        assert test.dataset.augmenter is None  # type: ignore[union-attr]
