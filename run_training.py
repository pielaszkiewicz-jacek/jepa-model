#!/usr/bin/env python3
"""
R-JEPA Stock Prediction — Training Script.

Usage:
    python run_training.py --config config/config.yaml
    python run_training.py --ticker AAPL --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockDataConfig, StockDataLoader, create_dataloaders
from models.r_jepa import RJEPA
from training.trainer import RJEPATrainer
from utils.metrics import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-JEPA for stock prediction")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seq_len", type=int, help="Sequence length")
    parser.add_argument("--pred_horizon", type=int, help="Prediction horizon")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension")
    parser.add_argument("--save_dir", type=str, help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cuda", "cpu"), help="Device")
    parser.add_argument("--conv_downsample", action="store_true",
                        help="Enable Conv1D downsampling (long sequences)")
    parser.add_argument("--use_transformer", action="store_true",
                        help="Use TransformerEncoder instead of BiLSTM")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Transformer attention heads")
    parser.add_argument("--sampling_interval", type=str,
                        help="Resample HF data to interval (e.g. 1s, 100ms)")
    parser.add_argument("--high_freq_features", action="store_true",
                        help="Enable HFT-specific features")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps (effective batch *= N)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "ticker": ("data", "tickers"),
        "start": ("data", "start_date"),
        "end": ("data", "end_date"),
        "epochs": ("training", "num_epochs"),
        "batch_size": ("data", "batch_size"),
        "lr": ("training", "learning_rate"),
        "seq_len": ("data", "sequence_length"),
        "pred_horizon": ("data", "prediction_horizon"),
        "latent_dim": ("model", "r_jepa", "latent_dim"),
        "save_dir": ("training", "save_dir"),
        "gradient_accumulation": ("training", "gradient_accumulation_steps"),
        "sampling_interval": ("data", "sampling_interval"),
    }
    for arg_name, keys in overrides.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            target = config
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = val

    # Boolean flags
    if args.conv_downsample:
        config.setdefault("model", {}).setdefault("r_jepa", {})["conv_downsample"] = True
    if args.use_transformer:
        config.setdefault("model", {}).setdefault("r_jepa", {})["use_transformer"] = True
    if args.nhead != 8:
        config.setdefault("model", {}).setdefault("r_jepa", {})["nhead"] = args.nhead
    if args.high_freq_features:
        config.setdefault("data", {})["use_high_freq_features"] = True

    return config


def get_device(device_str: str) -> str:
    match device_str:
        case "auto":
            return "cuda" if __import__("torch").cuda.is_available() else "cpu"
        case "cuda":
            return "cuda"
        case _:
            return "cpu"


def main() -> None:
    args = parse_args()
    config = override_config(load_config(args.config), args)
    device = get_device(args.device)

    print("=" * 60)
    print("R-JEPA Stock Prediction — Training")
    print("=" * 60)
    print(f"  Tickers: {config['data']['tickers']}")
    print(f"  Date range: {config['data']['start_date']} → {config['data']['end_date']}")
    print(f"  Sequence length: {config['data']['sequence_length']}")
    print(f"  Prediction horizon: {config['data']['prediction_horizon']}")
    print(f"  Latent dim: {config['model']['r_jepa']['latent_dim']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}\n")

    # ── Step 1: Data ────────────────────────────────────────────
    print("Step 1: Loading stock data...")
    data_cfg = StockDataConfig(
        tickers=tuple(config["data"]["tickers"]),
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        sequence_length=config["data"]["sequence_length"],
        prediction_horizon=config["data"]["prediction_horizon"],
        features=tuple(config["data"]["features"]),
        normalize=config["data"].get("normalize", True),
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
    )
    loader = StockDataLoader(data_cfg)
    data = loader.preprocess()
    n_features = data.shape[1]
    print(f"  Total data shape: {data.shape}")
    print(f"  Feature names: {loader.get_feature_names()}\n")

    # ── Step 2: Dataloaders ────────────────────────────────────
    print("Step 2: Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data, data_cfg, batch_size=config["data"]["batch_size"], num_workers=2,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")

    # ── Step 3: Model ───────────────────────────────────────────
    print("Step 3: Initializing R-JEPA model...")
    mc = config["model"]["r_jepa"]
    model = RJEPA(
        input_dim=n_features,
        encoder_hidden_dim=mc["encoder_hidden_dim"],
        encoder_num_layers=mc["encoder_num_layers"],
        latent_dim=mc["latent_dim"],
        predictor_hidden_dim=mc["predictor_hidden_dim"],
        predictor_num_layers=mc["predictor_num_layers"],
        predictor_dropout=mc.get("predictor_dropout", 0.1),
        decoder_hidden_dim=mc.get("decoder_hidden_dim", 64),
        decoder_output_dim=n_features,
        momentum_tau=mc.get("momentum_encoder_tau", 0.996),
        conv_downsample=mc.get("conv_downsample", False),
        use_transformer=mc.get("use_transformer", False),
        nhead=mc.get("nhead", 8),
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    if mc.get("conv_downsample"):
        print("  Encoder: Conv1D downsampling enabled")
    if mc.get("use_transformer"):
        print(f"  Encoder: Transformer (nhead={mc.get('nhead', 8)})")
    print()

    # ── Step 4: Train ───────────────────────────────────────────
    print("Step 4: Training...")
    trainer = RJEPATrainer(model, train_loader, val_loader, config, device=device)
    trainer.train()

    # ── Step 5: Plot ────────────────────────────────────────────
    print("\nStep 5: Generating training plots...")
    plot_path = os.path.join(config["training"]["save_dir"], "training_history.png")
    plot_training_history(
        trainer.train_losses,
        trainer.val_losses,
        trainer.learning_rates,
        save_path=plot_path,
        show=False,
    )

    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"  Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"  Checkpoints: {config['training']['save_dir']}/")
    print(f"  Plot: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
