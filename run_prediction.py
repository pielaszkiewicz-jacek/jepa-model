#!/usr/bin/env python3
"""
R-JEPA Stock Prediction — Inference / Prediction Script.

Usage:
    python run_prediction.py --checkpoint checkpoints/checkpoint_best.pt
    python run_prediction.py --checkpoint checkpoints/checkpoint_best.pt --ticker AAPL --horizon 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockDataConfig, StockDataLoader
from models.r_jepa import RJEPA
from utils.metrics import plot_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict stock prices with R-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--ticker", type=str, help="Stock ticker to predict")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon (overrides config)")
    parser.add_argument("--ensemble", action="store_true", default=True,
                        help="Use ensemble predictions with confidence intervals")
    parser.add_argument("--ensemble_size", type=int, default=10,
                        help="Number of ensemble trajectories")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Path to save predictions CSV")
    parser.add_argument("--plot", type=str, default="predictions.png",
                        help="Path to save prediction plot")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cuda", "cpu"), help="Device")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    match device_str:
        case "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case "cuda":
            return torch.device("cuda")
        case _:
            return torch.device("cpu")


def load_model(
    checkpoint_path: str, config: dict, device: torch.device,
) -> tuple[RJEPA, dict | None]:
    """
    Load model checkpoint and optional StandardScaler data.

    Returns:
        Tuple of (model, scaler_data), where scaler_data is a dict
        with ``"mean"`` and ``"scale"`` keys (lists), or ``None`` if
        the checkpoint does not contain scaler information.
    """
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Restore the full training config from checkpoint if available,
    # otherwise fall back to the YAML config (which may have fewer fields).
    full_config = ckpt.get("config", config)
    mc = full_config.get("model", {}).get("r_jepa", {})

    # Determine input dimension from the training data config.
    # Use the stored config's feature list + the known engineered features count.
    dc = full_config.get("data", {})
    base_features = dc.get("features", ["Open", "High", "Low", "Close", "Volume"])
    # Count engineered features by simulating what StockDataLoader._add_technical_features adds:
    # standard: Returns, Log_Returns, High_Low_Ratio, Close_Open_Ratio,
    #           MA_5, MA_20, MA_Ratio_5_20, Volatility_5, Volatility_20,
    #           Volume_MA_5, Volume_Ratio, RSI_14  → 12 engineered features
    engineerd_count = 12
    use_hf = dc.get("use_high_freq_features", False)
    if use_hf:
        # HF features: Spread, Mid_Price (2 if Bid/Ask exist),
        # Volume_Imbalance, Trade_Intensity, Micro_Volatility,
        # Amihud_Illiq, Tick_Momentum → up to 7 extra
        engineerd_count += 7
    n_features = len(base_features) + engineerd_count

    model = RJEPA(
        input_dim=n_features,
        encoder_hidden_dim=mc.get("encoder_hidden_dim", 128),
        encoder_num_layers=mc.get("encoder_num_layers", 2),
        latent_dim=mc.get("latent_dim", 64),
        predictor_hidden_dim=mc.get("predictor_hidden_dim", 128),
        predictor_num_layers=mc.get("predictor_num_layers", 2),
        predictor_dropout=mc.get("predictor_dropout", 0.1),
        decoder_hidden_dim=mc.get("decoder_hidden_dim", 64),
        decoder_output_dim=n_features,
        momentum_tau=mc.get("momentum_encoder_tau", 0.996),
        conv_downsample=mc.get("conv_downsample", False),
        use_transformer=mc.get("use_transformer", False),
        nhead=mc.get("nhead", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Extract saved scaler data if present
    scaler_data: dict | None = ckpt.get("scaler_data")
    if scaler_data is not None:
        print(f"  StandardScaler loaded from checkpoint "
              f"({len(scaler_data['mean'])} features)")
    else:
        print("  WARNING: No StandardScaler found in checkpoint — "
              "inverse transforms may be incorrect")

    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")
    return model, scaler_data


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    horizon = args.horizon or config["data"].get("prediction_horizon", 5)

    print("=" * 60)
    print("R-JEPA Stock Prediction — Inference")
    print("=" * 60)
    print(f"  Ticker: {config['data']['tickers']}")
    print(f"  Horizon: {horizon} days")
    print(f"  Ensemble: {args.ensemble}\n")

    model, scaler_data = load_model(args.checkpoint, config, device)

    # Load & preprocess data
    print("\nLoading recent data...")
    dc = config["data"]
    tickers_val = dc["tickers"]
    if isinstance(tickers_val, str):
        tickers_val = [tickers_val]
    data_cfg = StockDataConfig(
        tickers=tuple(tickers_val),
        start_date=dc["start_date"],
        end_date=dc["end_date"],
        sequence_length=dc["sequence_length"],
        prediction_horizon=horizon,
        features=tuple(dc["features"]),
        normalize=dc.get("normalize", True),
        sampling_interval=dc.get("sampling_interval"),
        use_high_freq_features=dc.get("use_high_freq_features", False),
        conv_downsample=config.get("model", {}).get("r_jepa", {}).get("conv_downsample", False),
        use_transformer_encoder=config.get("model", {}).get("r_jepa", {}).get("use_transformer", False),
    )
    loader = StockDataLoader(data_cfg)
    all_data = loader.preprocess()
    seq_len = dc["sequence_length"]

    # Override the data loader's scaler with checkpoint's scaler (if available)
    # so that inverse_transform() uses the same normalisation as training.
    if scaler_data is not None:
        ckpt_scaler = StandardScaler()
        ckpt_scaler.mean_ = np.array(scaler_data["mean"], dtype=np.float64)
        scale_arr = np.array(scaler_data["scale"], dtype=np.float64)
        ckpt_scaler.scale_ = scale_arr
        ckpt_scaler.var_ = scale_arr ** 2  # type: ignore[operator]
        loader._scaler = ckpt_scaler  # type: ignore[union-attr]
        print("  Using StandardScaler from checkpoint for inverse transforms")

    # Prepare context: last seq_len samples
    context = torch.from_numpy(all_data[-seq_len:]).float().unsqueeze(0).to(device)
    print(f"  Context shape: {context.shape}")

    # Generate predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        if args.ensemble:
            output = model.predict(
                context=context,
                prediction_horizon=horizon,
                return_ensemble=True,
                ensemble_size=args.ensemble_size,
                noise_scale=0.05,
            )
            preds = output["predictions"].cpu().numpy()[0]
            upper = output["upper_bound"].cpu().numpy()[0]
            lower = output["lower_bound"].cpu().numpy()[0]
            print(f"  Mean confidence: {output['confidence'].mean().item():.3f}")
        else:
            output = model.predict(context=context, prediction_horizon=horizon)
            preds = output["predictions"].cpu().numpy()[0]
            upper = lower = None

    # Inverse transform
    n_features_data = all_data.shape[1]
    if preds.shape[1] < n_features_data:
        padded = np.zeros((preds.shape[0], n_features_data))
        padded[:, : preds.shape[1]] = preds
        preds_orig = loader.inverse_transform(padded)[:, : preds.shape[1]]
    else:
        preds_orig = loader.inverse_transform(preds)

    # Inverse-transform confidence bounds as well (same padding scheme)
    if upper is not None and lower is not None:
        if upper.shape[1] < n_features_data:
            upper_padded = np.zeros((upper.shape[0], n_features_data))
            upper_padded[:, : upper.shape[1]] = upper
            upper_orig = loader.inverse_transform(upper_padded)[:, : upper.shape[1]]
        else:
            upper_orig = loader.inverse_transform(upper)

        if lower.shape[1] < n_features_data:
            lower_padded = np.zeros((lower.shape[0], n_features_data))
            lower_padded[:, : lower.shape[1]] = lower
            lower_orig = loader.inverse_transform(lower_padded)[:, : lower.shape[1]]
        else:
            lower_orig = loader.inverse_transform(lower)
    else:
        upper_orig = lower_orig = None

    feature_names = loader.get_feature_names()

    print(f"\nPredictions (next {horizon} days):" + "\n" + "-" * 40)
    for i in range(horizon):
        vals = [f"{preds_orig[i, j]:.2f}" for j in range(min(5, preds_orig.shape[1]))]
        print(f"  Day {i+1}: Open={vals[0]}, High={vals[1]}, Low={vals[2]}, Close={vals[3]}, Vol={vals[4]}")

    # Plot
    print("\nGenerating plot...")
    hist_orig = loader.inverse_transform(all_data)
    plot_predictions(
        historical=hist_orig,
        predictions=preds_orig,
        targets=None,
        upper_bound=upper_orig,
        lower_bound=lower_orig,
        feature_names=feature_names[:4],
        title=f"{config['data']['tickers'][0]} — R-JEPA Prediction",
        save_path=args.plot,
        show=False,
    )

    # Save CSV
    print("Saving predictions...")
    records: dict[str, list] = {"Day": list(range(1, horizon + 1))}
    for j, name in enumerate(feature_names[: preds_orig.shape[1]]):
        records[f"{name}_predicted"] = list(preds_orig[:, j])
        if upper is not None and lower is not None:
            records[f"{name}_upper"] = list(upper[:, j])
            records[f"{name}_lower"] = list(lower[:, j])
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"  Saved to {args.output}")

    print(f"\n{'=' * 60}")
    print("Prediction completed!")
    print(f"  Plot: {args.plot}")
    print(f"  CSV:  {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
