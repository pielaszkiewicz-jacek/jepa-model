"""Shared fixtures for R-JEPA tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from models.decoder import StockDecoder
from models.encoder import TimeSeriesEncoder
from models.predictor import RecurrentPredictor
from models.r_jepa import RJEPA


# ── Seeds ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Reset random seeds before every test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


# ── Synthetic data ───────────────────────────────────────────────────────


@pytest.fixture
def n_features() -> int:
    return 5


@pytest.fixture
def seq_len() -> int:
    return 60


@pytest.fixture
def pred_horizon() -> int:
    return 5


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def synthetic_data(n_features: int, seq_len: int) -> np.ndarray:
    """Standard-normal random time series, shape ``(total_len, n_features)``."""
    total_len = seq_len + 1000  # enough for train/val/test partitions
    return np.random.randn(total_len, n_features).astype(np.float32)


@pytest.fixture
def context_tensor(batch_size: int, seq_len: int, n_features: int) -> Tensor:
    return torch.randn(batch_size, seq_len, n_features)


@pytest.fixture
def target_tensor(batch_size: int, pred_horizon: int, n_features: int) -> Tensor:
    return torch.randn(batch_size, pred_horizon, n_features)


# ── Small model fixtures ─────────────────────────────────────────────────


@pytest.fixture
def small_encoder(n_features: int) -> TimeSeriesEncoder:
    return TimeSeriesEncoder(
        input_dim=n_features,
        hidden_dim=16,
        num_layers=1,
        latent_dim=8,
    )


@pytest.fixture
def small_predictor() -> RecurrentPredictor:
    return RecurrentPredictor(
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def small_decoder() -> StockDecoder:
    return StockDecoder(
        latent_dim=8,
        hidden_dim=16,
        output_dim=5,
        num_layers=2,
    )


@pytest.fixture
def small_r_jepa(n_features: int) -> RJEPA:
    return RJEPA(
        input_dim=n_features,
        encoder_hidden_dim=16,
        encoder_num_layers=1,
        latent_dim=8,
        predictor_hidden_dim=16,
        predictor_num_layers=1,
        predictor_dropout=0.0,
        decoder_hidden_dim=16,
    )
