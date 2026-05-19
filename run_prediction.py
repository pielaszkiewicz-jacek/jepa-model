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
) -> RJEPA:
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    mc = config.get("model", {}).get("r_jepa", {})
    n_features = len(config.get("data", {}).get("features", ["Open", "High", "Low", "Close", "Volume"]))
    n_features += 8  # engineered features

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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")
    return model


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

    model = load_model(args.checkpoint, config, device)

    # Load & preprocess data
    print("\nLoading recent data...")
    data_cfg = StockDataConfig(
        tickers=tuple(config["data"]["tickers"]),
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        sequence_length=config["data"]["sequence_length"],
        prediction_horizon=horizon,
        features=tuple(config["data"]["features"]),
        normalize=config["data"].get("normalize", True),
    )
    loader = StockDataLoader(data_cfg)
    all_data = loader.preprocess()
    seq_len = config["data"]["sequence_length"]

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
        upper_bound=upper,
        lower_bound=lower,
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
