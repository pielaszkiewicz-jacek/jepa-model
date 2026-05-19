"""
Walk-Forward Validation for R-JEPA Time Series Models.

Unlike standard k-fold cross-validation, walk-forward validation preserves
temporal order: each fold trains on *past* data and validates on *future*
data, preventing look-ahead bias. This is the standard approach for
evaluating financial time series models.

Usage::

    validator = WalkForwardValidator(
        data=normalized_data,
        config=walk_config,
        trainer_builder=trainer_builder,
        verbose=True,
    )
    results = validator.run()

Or from the CLI::

    python run_training.py --config config.yaml --walk_forward --wf_splits 5
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.stock_data import StockDataConfig, StockDataset
from models.r_jepa import RJEPA
from training.trainer import RJEPATrainer


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Uses an expanding-window strategy:
        Fold 0: train [0, p1)        → validate [p1, p2)
        Fold 1: train [0, p2)        → validate [p2, p3)
        Fold 2: train [0, p3)        → validate [p3, p4)
        ...

    Attributes:
        n_splits: Number of validation windows (folds).
        min_train_fraction: Minimum fraction of total data used for training
            in the earliest fold. The remaining data is split into equally-sized
            validation windows. Default 0.4 ensures fold 0 has 40% train / 60% split
            into ``n_splits`` validation chunks.
        batch_size: Batch size for all folds.
        num_workers: Number of DataLoader workers.
        verbose: Print per-fold progress.
    """

    n_splits: int = 5
    min_train_fraction: float = 0.4
    early_stopping_patience: int = 10
    batch_size: int = 32
    num_workers: int = 2
    verbose: bool = True


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""

    fold: int
    """Zero-based fold index."""

    train_start: int
    """Start index of training data (inclusive)."""

    train_end: int
    """End index of training data (exclusive)."""

    val_start: int
    """Start index of validation data (inclusive)."""

    val_end: int
    """End index of validation data (exclusive)."""

    best_val_loss: float
    """Best validation loss achieved during training."""

    epochs_trained: int
    """Number of epochs actually trained (may be < max due to early stopping)."""

    train_time_sec: float
    """Wall-clock time for training this fold."""

    fold_config: dict[str, Any] = field(default_factory=dict)
    """Configuration snapshot for this fold."""


class WalkForwardValidator:
    """
    Walk-forward validator for time series models.

    Builds a fresh :class:`RJEPA` model and :class:`RJEPATrainer` for each fold,
    trains with early stopping, and collects per-fold metrics.

    Args:
        data: Normalized feature array of shape ``(total_timesteps, n_features)``.
        walk_config: Walk-forward validation parameters.
        model_config: Dictionary of model hyperparameters (same keys as
            ``config["model"]["r_jepa"]`` in the YAML).
        data_config: Data configuration used to derive sequence length and
            prediction horizon.
        training_config: Training configuration dict (num_epochs, learning_rate, etc.).
        trainer_builder: Optional factory ``(RJEPA, DataLoader, DataLoader) -> RJEPATrainer``.
            When absent, the validator constructs the trainer automatically using
            ``RJEPATrainer(model, train_loader, val_loader, full_config)``.
        full_config: Full configuration dict passed to ``RJEPATrainer``. Required when
            ``trainer_builder`` is *not* provided.
        device: Torch device. Auto-detected if ``None``.
    """

    def __init__(
        self,
        data: np.ndarray,
        walk_config: WalkForwardConfig,
        model_config: dict[str, Any],
        data_config: StockDataConfig,
        training_config: dict[str, Any],
        trainer_builder: Callable[[RJEPA, DataLoader, DataLoader], RJEPATrainer] | None = None,
        full_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.data = data
        self.walk_config = walk_config
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.trainer_builder = trainer_builder
        self.full_config = full_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Compute fold boundaries
        self.folds: list[dict[str, int]] = self._compute_folds()

        self.results: list[WalkForwardResult] = []

    # ── Public API ────────────────────────────────────────────────────

    def run(self) -> list[WalkForwardResult]:
        """Execute walk-forward validation across all folds.

        Returns:
            List of :class:`WalkForwardResult` — one per fold, in chronological order.
        """
        self.results = []
        n_features = self.data.shape[1]

        for i, fold in enumerate(self.folds):
            if self.walk_config.verbose:
                print(f"\n{'─' * 60}")
                print(f"Walk-Forward Fold {i + 1}/{self.walk_config.n_splits}")
                print(f"{'─' * 60}")
                train_pct = (fold["train_end"] - fold["train_start"]) / len(self.data) * 100
                val_pct = (fold["val_end"] - fold["val_start"]) / len(self.data) * 100
                print(f"  Train: [{fold['train_start']}:{fold['train_end']}] "
                      f"({train_pct:.0f}% of data)")
                print(f"  Val:   [{fold['val_start']}:{fold['val_end']}] "
                      f"({val_pct:.0f}% of data)")

            # ── 1. Slice data for this fold ──────────────────────
            train_data = self.data[fold["train_start"] : fold["train_end"]]
            val_data = self.data[fold["val_start"] : fold["val_end"]]

            if len(train_data) < self.data_config.sequence_length + self.data_config.prediction_horizon:
                msg = (
                    f"Fold {i}: training data too short "
                    f"({len(train_data)} rows, need ≥ "
                    f"{self.data_config.sequence_length + self.data_config.prediction_horizon})"
                )
                raise ValueError(msg)

            # ── 2. Create DataLoaders ────────────────────────────
            common_dataset_kwargs = dict(
                sequence_length=self.data_config.sequence_length,
                prediction_horizon=self.data_config.prediction_horizon,
            )

            train_dataset = StockDataset(train_data, augmenter=None, **common_dataset_kwargs)
            val_dataset = StockDataset(val_data, augmenter=None, **common_dataset_kwargs)

            loader_kwargs: dict = dict(
                batch_size=self.walk_config.batch_size,
                num_workers=self.walk_config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

            if self.walk_config.verbose:
                print(f"  Train batches: {len(train_loader)}")
                print(f"  Val batches:   {len(val_loader)}")

            # ── 3. Build model ───────────────────────────────────
            model = RJEPA(
                input_dim=n_features,
                encoder_hidden_dim=self.model_config.get("encoder_hidden_dim", 128),
                encoder_num_layers=self.model_config.get("encoder_num_layers", 2),
                latent_dim=self.model_config.get("latent_dim", 64),
                predictor_hidden_dim=self.model_config.get("predictor_hidden_dim", 128),
                predictor_num_layers=self.model_config.get("predictor_num_layers", 2),
                predictor_dropout=self.model_config.get("predictor_dropout", 0.1),
                decoder_hidden_dim=self.model_config.get("decoder_hidden_dim", 64),
                decoder_output_dim=n_features,
                momentum_tau=self.model_config.get("momentum_encoder_tau", 0.996),
                conv_downsample=self.model_config.get("conv_downsample", False),
                use_transformer=self.model_config.get("use_transformer", False),
                nhead=self.model_config.get("nhead", 8),
            )

            # ── 4. Build trainer ─────────────────────────────────
            if self.trainer_builder is not None:
                trainer = self.trainer_builder(model, train_loader, val_loader)
            else:
                if self.full_config is None:
                    msg = "Either trainer_builder or full_config must be provided"
                    raise ValueError(msg)
                # Override num_epochs and patience with fold-specific values
                fold_config = copy.deepcopy(self.full_config)
                fold_config.setdefault("training", {}).setdefault("num_epochs",
                    self.training_config.get("num_epochs", 100))
                fold_config.setdefault("training", {}).setdefault("patience",
                    self.walk_config.early_stopping_patience)
                trainer = RJEPATrainer(
                    model, train_loader, val_loader, fold_config, device=self.device,
                )

            # ── 5. Train ─────────────────────────────────────────
            t0 = time.perf_counter()
            trainer.train()
            elapsed = time.perf_counter() - t0

            # ── 6. Record result ─────────────────────────────────
            result = WalkForwardResult(
                fold=i,
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                val_start=fold["val_start"],
                val_end=fold["val_end"],
                best_val_loss=trainer.best_val_loss,
                epochs_trained=trainer.current_epoch,
                train_time_sec=elapsed,
                fold_config={
                    "n_features": n_features,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                },
            )
            self.results.append(result)

            if self.walk_config.verbose:
                print(f"  ✓ Fold {i + 1} complete — "
                      f"best val loss: {result.best_val_loss:.6f}, "
                      f"epochs: {result.epochs_trained}, "
                      f"time: {elapsed:.1f}s")

        # ── Summary ──────────────────────────────────────────────
        if self.walk_config.verbose:
            self._print_summary()

        return self.results

    def summary(self) -> dict[str, Any]:
        """Aggregate results across all folds into a single summary dict."""
        if not self.results:
            return {}

        val_losses = [r.best_val_loss for r in self.results]
        epochs = [r.epochs_trained for r in self.results]
        times = [r.train_time_sec for r in self.results]

        return {
            "n_splits": self.walk_config.n_splits,
            "mean_val_loss": float(np.mean(val_losses)),
            "std_val_loss": float(np.std(val_losses)),
            "min_val_loss": float(np.min(val_losses)),
            "max_val_loss": float(np.max(val_losses)),
            "val_losses_per_fold": val_losses,
            "mean_epochs": float(np.mean(epochs)),
            "total_time_sec": float(np.sum(times)),
            "mean_time_per_fold_sec": float(np.mean(times)),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_folds(self) -> list[dict[str, int]]:
        """Compute expanding-window fold boundaries.

        The data is partitioned into:
            1. An initial training block (``min_train_fraction`` of total data).
            2. ``n_splits`` equal-sized validation blocks laid out sequentially.

        Fold i trains on ``[0, val_start_i)`` and validates on ``[val_start_i, val_end_i)``.

        Returns:
            List of dicts with keys ``train_start``, ``train_end``, ``val_start``, ``val_end``.
        """
        total = len(self.data)
        n = self.walk_config.n_splits
        min_train = int(total * self.walk_config.min_train_fraction)

        # Remaining data after reserving the minimum training block
        remaining = total - min_train
        val_window = remaining // n  # floor — last fold may be slightly larger

        if val_window < self.data_config.sequence_length + self.data_config.prediction_horizon:
            msg = (
                f"Validation window ({val_window} rows) is smaller than the minimum "
                f"required ({self.data_config.sequence_length + self.data_config.prediction_horizon}). "
                f"Reduce n_splits or increase min_train_fraction."
            )
            raise ValueError(msg)

        folds: list[dict[str, int]] = []
        for i in range(n):
            val_start = min_train + i * val_window
            val_end = min_train + (i + 1) * val_window if i < n - 1 else total

            folds.append({
                "train_start": 0,
                "train_end": val_start,
                "val_start": val_start,
                "val_end": val_end,
            })

        return folds

    def _print_summary(self) -> None:
        """Print an aggregate summary of all folds."""
        s = self.summary()
        if not s:
            return

        print(f"\n{'═' * 60}")
        print("Walk-Forward Validation — Summary")
        print(f"{'═' * 60}")
        print(f"  Folds:              {s['n_splits']}")
        print(f"  Mean val loss:      {s['mean_val_loss']:.6f} ± {s['std_val_loss']:.6f}")
        print(f"  Min val loss:       {s['min_val_loss']:.6f}")
        print(f"  Max val loss:       {s['max_val_loss']:.6f}")
        print(f"  Mean epochs/fold:   {s['mean_epochs']:.1f}")
        print(f"  Total time:         {s['total_time_sec']:.1f}s")
        print(f"  Mean time/fold:     {s['mean_time_per_fold_sec']:.1f}s")
        print(f"{'═' * 60}")
