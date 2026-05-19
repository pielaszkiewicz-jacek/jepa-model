"""
Stock Data Loader and Dataset for R-JEPA model.
Downloads historical stock data and creates sequences for training/prediction.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from data.augmentation import (
    Compose,
    GaussianNoise,
    MagnitudeWarping,
    TimeWarping,
    WindowSlice,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class StockDataConfig:
    """Configuration for stock data loading.

    Supports both daily and high-frequency (intraday/tick) data.

    Attributes:
        tickers: Stock ticker symbols.
        start_date: Start date (YYYY-MM-DD) for daily; also used as filter for HF.
        end_date: End date (YYYY-MM-DD).
        sequence_length: Number of past time steps (context window).
        prediction_horizon: Number of future time steps to predict.
        features: Column names to extract from raw data.
        normalize: Whether to apply StandardScaler normalization.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        sampling_interval: Resampling interval for HF data, e.g. "1s", "100ms", "1min".
            If None, data is assumed to be daily (no resampling).
        use_high_freq_features: Whether to compute HFT-specific features.
        conv_downsample: Whether encoder should use Conv1D downsampling layers.
        use_transformer_encoder: Whether to use TransformerEncoder instead of BiLSTM.
    """

    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    sequence_length: int = 60
    prediction_horizon: int = 5
    features: tuple[str, ...] = field(
        default_factory=lambda: ("Open", "High", "Low", "Close", "Volume")
    )
    normalize: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    sampling_interval: str | None = None
    use_high_freq_features: bool = False
    conv_downsample: bool = False
    use_transformer_encoder: bool = False
    # Data augmentation (context-only, training-only)
    aug_noise_std: float = 0.0
    aug_magnitude_sigma: float = 0.0
    aug_time_warp_sigma: float = 0.0
    aug_window_slice_ratio: float = 0.0


class StockDataLoader:
    """
    Downloads and preprocesses stock market data from Yahoo Finance.
    Supports multiple tickers, daily and high-frequency data, feature engineering.

    For high-frequency data, call `load_high_freq_data(csv_path)` instead of
    the default Yahoo Finance download, providing a CSV with columns:
        timestamp, Open, High, Low, Close, Volume[, Bid, Ask]
    """

    def __init__(self, config: StockDataConfig) -> None:
        self.config = config
        self._scaler = StandardScaler()
        self.data: dict[str, pd.DataFrame] = {}
        self.processed_data: dict[str, np.ndarray] = {}
        self._download_data()

    # ── Public API ──────────────────────────────────────────────

    def preprocess(self) -> np.ndarray:
        """
        Preprocess data: resample if HF, handle NaN, normalize, combine tickers.
        Returns normalized feature matrix.

        Normalization uses a SINGLE StandardScaler fitted on the COMBINED
        training portion of ALL tickers, ensuring consistent scaling across
        the portfolio and a single correct inverse_transform.
        """
        all_clean: list[np.ndarray] = []

        # Step 1: Resample (if needed) and clean each ticker's data
        for ticker, df in self.data.items():
            if self.config.sampling_interval is not None:
                df = self.resample_to_interval(df, self.config.sampling_interval)

            df_clean = df.dropna()
            self.processed_data[ticker] = df_clean.values
            all_clean.append(df_clean.values)
            print(f"  Cleaned {ticker}: {df_clean.shape[1]} features, "
                  f"{df_clean.shape[0]} time steps")

        # Step 2: Fit scaler ONCE on the combined training portion from ALL tickers
        if self.config.normalize:
            train_slices: list[np.ndarray] = []
            for arr in all_clean:
                train_size = int(len(arr) * self.config.train_split)
                train_slices.append(arr[:train_size])
            train_combined = np.concatenate(train_slices, axis=0)
            self._scaler.fit(train_combined)
            print(f"  Scaler fitted on combined training data: "
                  f"{train_combined.shape[0]} rows from {len(all_clean)} ticker(s)")

        # Step 3: Transform each ticker's data with the shared scaler
        scaled_all: list[np.ndarray] = []
        for ticker, arr in zip(self.data, all_clean):
            if self.config.normalize:
                scaled = self._scaler.transform(arr)
            else:
                scaled = arr
            self.processed_data[ticker] = scaled
            scaled_all.append(scaled)
            print(f"  Preprocessed {ticker}: {scaled.shape[1]} features, "
                  f"{scaled.shape[0]} time steps")

        return np.concatenate(scaled_all, axis=0)

    def get_feature_names(self) -> list[str]:
        """Get names of all features including engineered ones."""
        if not self.data:
            return []
        return list(next(iter(self.data.values())).columns)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        return self._scaler.inverse_transform(data)

    # ── Internal helpers ────────────────────────────────────────

    def load_high_freq_data(self, csv_path: str, ticker: str | None = None) -> None:
        """Load high-frequency tick data from a CSV file and add features.

        Expected CSV columns (at minimum):
            timestamp, Open, High, Low, Close, Volume
        Optional:
            Bid, Ask  (for spread computation)

        Args:
            csv_path: Path to CSV file with high-frequency data.
            ticker: Ticker symbol (defaults to filename stem).
        """
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Keep only configured features that exist
        available = [f for f in self.config.features if f in df.columns]
        df = df[available]

        # Add technical features (standard + HF-specific if configured)
        df = self._add_technical_features(df)

        ticker = ticker or Path(csv_path).stem
        self.data[ticker] = df
        print(f"  Loaded {len(df)} rows for {ticker} from {csv_path}")

    def resample_to_interval(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample irregular tick data to regular time intervals.

        Uses last observed value for OHLC, sum for Volume.
        For high-frequency data where Bid/Ask are available, computes
        mid-price before resampling.

        Args:
            df: DataFrame with DatetimeIndex.
            interval: Resampling interval, e.g. "1s", "100ms", "1min", "1h".

        Returns:
            DataFrame resampled to regular intervals with forward-fill.
        """
        numeric = df.select_dtypes(include=[np.number])

        # Define aggregation rules per known column type
        ohlc_cols = {"Open", "High", "Low", "Close"}
        agg_rules: dict[str, str] = {}
        for col in numeric.columns:
            if col in ohlc_cols:
                agg_rules[col] = "last"
            elif col == "Volume":
                agg_rules[col] = "sum"
            else:
                # Bid, Ask, Spread, and any engineered columns → last observed
                agg_rules[col] = "last"

        resampled = numeric.resample(interval).agg(agg_rules)

        # Forward-fill short gaps, then drop any remaining NaN
        resampled = resampled.ffill(limit=5).dropna()
        return resampled

    def _download_data(self) -> None:
        """Download historical stock data for all configured tickers."""
        for ticker in self.config.tickers:
            print(f"Downloading data for {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=self.config.start_date,
                end=self.config.end_date,
            )
            if df.empty:
                msg = f"No data found for ticker {ticker}"
                raise ValueError(msg)

            available = [f for f in self.config.features if f in df.columns]
            df = df[available]
            df = self._add_technical_features(df)
            self.data[ticker] = df
            print(f"  Loaded {len(df)} rows for {ticker}")

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to enrich feature space.

        Always adds standard features (returns, MAs, RSI). If the config
        has ``use_high_freq_features=True``, also adds HFT-specific features
        such as spread, volume imbalance, trade intensity, and micro-volatility.
        """
        df = df.copy()

        # ── Standard daily features ────────────────────────────────
        # Price-based features
        df["Returns"] = df["Close"].pct_change()
        # Log_Returns: guard against zero/negative prices
        shifted_close = df["Close"].shift(1)
        df["Log_Returns"] = np.where(
            shifted_close > 0,
            np.log(df["Close"] / shifted_close),
            0.0,
        )
        df["High_Low_Ratio"] = df["High"] / df["Low"]
        df["Close_Open_Ratio"] = df["Close"] / df["Open"]

        # Moving averages
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_Ratio_5_20"] = df["MA_5"] / df["MA_20"]

        # Volatility
        df["Volatility_5"] = df["Returns"].rolling(window=5).std()
        df["Volatility_20"] = df["Returns"].rolling(window=20).std()

        # Volume-based features
        df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()
        # Guard against zero MA_5 for Volume_Ratio
        volume_ma_5_safe = df["Volume_MA_5"].replace(0, np.nan)
        df["Volume_Ratio"] = df["Volume"] / volume_ma_5_safe

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Guard against zero loss (RSI = 100 when no losses)
        loss_safe = loss.replace(0, np.nan)
        rs = gain / loss_safe
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # ── High-frequency features (optional) ─────────────────────
        if self.config.use_high_freq_features:
            df = self._add_high_freq_features(df)

        return df

    def _add_high_freq_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to high-frequency / intraday data.

        These capture microstructure effects such as bid-ask spread,
        order-flow imbalance, and intra-interval volatility.
        """
        eps = 1e-10  # numerical stability floor

        # Spread (if Bid/Ask columns exist)
        if "Bid" in df.columns and "Ask" in df.columns:
            mid = (df["Ask"] + df["Bid"]) / 2
            df["Spread"] = (df["Ask"] - df["Bid"]) / mid.where(mid > 0, np.nan)
            df["Mid_Price"] = mid

        # Volume Imbalance — proxy for order-flow pressure
        # Positive = buying pressure, Negative = selling pressure
        volume_diff = df["Volume"].diff()
        df["Volume_Imbalance"] = volume_diff / (df["Volume"] + eps)

        # Trade Intensity — volume per time step
        volume_ma_10 = df["Volume"].rolling(10).mean()
        df["Trade_Intensity"] = df["Volume"] / (volume_ma_10 + eps)

        # Micro-Volatility — high-frequency volatility (intra-interval)
        df["Micro_Volatility"] = (
            (df["High"] - df["Low"]) / (df["Close"] + eps)
        )

        # Amihud Illiquidity Ratio — price impact per unit volume
        # Higher values indicate lower liquidity
        abs_return = df["Returns"].abs()
        df["Amihud_Illiq"] = abs_return / (df["Volume"] + eps)

        # Tick-level momentum — acceleration of returns
        df["Tick_Momentum"] = df["Returns"].diff()

        return df


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock time series.
    Creates context-target pairs for R-JEPA training.

    For each sample:
    - context: sequence of past observations [seq_len, n_features]
    - target: sequence of future observations [pred_horizon, n_features]
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_horizon: int,
        stride: int = 1,
        augmenter: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.augmenter = augmenter

        total_len = len(data)
        self.valid_indices = list(
            range(0, total_len - sequence_length - prediction_horizon + 1, stride)
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start_idx = self.valid_indices[idx]
        end_context = start_idx + self.sequence_length
        end_target = end_context + self.prediction_horizon

        context = self.data[start_idx:end_context]  # [seq_len, n_features]
        target = self.data[end_context:end_target]  # [pred_horizon, n_features]

        # Augmentation: only applied to context (not target), so the
        # prediction task becomes harder while the ground-truth future
        # remains unchanged.
        if self.augmenter is not None:
            context = self.augmenter(context)

        return context, target


def _build_augmenter(config: StockDataConfig) -> Compose | None:
    """Build augmentation pipeline from config, or None if all disabled."""
    transforms: list[Callable[[Tensor], Tensor]] = []

    if config.aug_noise_std > 0:
        transforms.append(GaussianNoise(std=config.aug_noise_std))
    if config.aug_magnitude_sigma > 0:
        transforms.append(
            MagnitudeWarping(sigma=config.aug_magnitude_sigma)
        )
    if config.aug_time_warp_sigma > 0:
        transforms.append(TimeWarping(sigma=config.aug_time_warp_sigma))
    if config.aug_window_slice_ratio > 0:
        transforms.append(
            WindowSlice(ratio=config.aug_window_slice_ratio)
        )

    return Compose(transforms) if transforms else None


def create_dataloaders(
    data: np.ndarray,
    config: StockDataConfig,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from preprocessed data.

    Args:
        data: Preprocessed numpy array of shape [n_timesteps, n_features]
        config: StockDataConfig instance
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n = len(data)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    common = dict(sequence_length=config.sequence_length,
                  prediction_horizon=config.prediction_horizon)

    # Build augmenter for training data (only if enabled in config)
    augmenter = _build_augmenter(config)

    train_dataset = StockDataset(train_data, augmenter=augmenter, **common)
    val_dataset = StockDataset(val_data, augmenter=None, **common)
    test_dataset = StockDataset(test_data, augmenter=None, **common)

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )
