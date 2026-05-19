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
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockDataConfig, StockDataLoader, create_dataloaders
from models.r_jepa import RJEPA
from training.trainer import RJEPATrainer
from training.walk_forward import WalkForwardConfig, WalkForwardValidator
from utils.experiment_tracking import ExperimentTracker
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

    # Data augmentation
    parser.add_argument("--aug_noise", type=float, default=0.0,
                        help="Gaussian noise std for context augmentation (default: 0 = off)")
    parser.add_argument("--aug_magnitude", type=float, default=0.0,
                        help="Magnitude warping sigma (default: 0 = off)")
    parser.add_argument("--aug_time_warp", type=float, default=0.0,
                        help="Time warping sigma (default: 0 = off)")
    parser.add_argument("--aug_window_slice", type=float, default=0.0,
                        help="Window slice ratio (default: 0 = off)")

    # Walk-forward validation
    parser.add_argument("--walk_forward", action="store_true",
                        help="Run walk-forward validation instead of single train/val split")
    parser.add_argument("--wf_splits", type=int, default=5,
                        help="Number of walk-forward validation folds (default: 5)")
    parser.add_argument("--wf_min_train", type=float, default=0.4,
                        help="Minimum training fraction for walk-forward (default: 0.4)")
    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (progressive prediction horizon)")
    parser.add_argument("--curriculum_initial_horizon", type=int, default=1,
                        help="Starting prediction horizon for curriculum (default: 1)")
    parser.add_argument("--curriculum_warmup_epochs", type=int, default=10,
                        help="Epochs to stay at initial horizon before increasing (default: 10)")
    # Experiment tracking (MLflow)
    parser.add_argument("--experiment", action="store_true",
                        help="Enable MLflow experiment tracking")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="MLflow experiment name (default: from config)")
    parser.add_argument("--tracking_uri", type=str, default=None,
                        help="MLflow tracking server URI (default: from config)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
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
        "aug_noise": ("data", "aug_noise_std"),
        "aug_magnitude": ("data", "aug_magnitude_sigma"),
        "aug_time_warp": ("data", "aug_time_warp_sigma"),
        "aug_window_slice": ("data", "aug_window_slice_ratio"),
        "curriculum_initial_horizon": ("training", "curriculum", "initial_horizon"),
        "curriculum_warmup_epochs": ("training", "curriculum", "warmup_epochs"),
        "curriculum_step_epochs": ("training", "curriculum", "step_epochs"),
    }
    for arg_name, keys in overrides.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            target = config
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = val

    # Handle --ticker separately: wrap string in a list so tuple() doesn't
    # split "AAPL" into ("A", "A", "P", "L").
    if args.ticker is not None:
        config.setdefault("data", {})["tickers"] = [args.ticker]

    # Boolean flags
    if args.conv_downsample:
        config.setdefault("model", {}).setdefault("r_jepa", {})["conv_downsample"] = True
    if args.use_transformer:
        config.setdefault("model", {}).setdefault("r_jepa", {})["use_transformer"] = True
    if args.nhead != 8:
        config.setdefault("model", {}).setdefault("r_jepa", {})["nhead"] = args.nhead
    if args.high_freq_features:
        config.setdefault("data", {})["use_high_freq_features"] = True

    # Curriculum learning flag — set 'enabled' in training.curriculum
    if args.curriculum:
        config.setdefault("training", {}).setdefault("curriculum", {})["enabled"] = True

    return config


def get_device(device_str: str) -> str:
    match device_str:
        case "auto":
            return "cuda" if __import__("torch").cuda.is_available() else "cpu"
        case "cuda":
            return "cuda"
        case _:
            return "cpu"


def _build_experiment_tracker(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> ExperimentTracker | None:
    """Build an ExperimentTracker from config and CLI overrides.

    Returns ``None`` when tracking is disabled.
    """
    experiment_cfg = config.get("experiment", {})
    enabled = (
        args.experiment
        or experiment_cfg.get("enabled", False)
    )
    if not enabled:
        return None

    # CLI overrides take precedence over config
    tracking_uri = args.tracking_uri or experiment_cfg.get("tracking_uri")
    experiment_name = args.experiment_name or experiment_cfg.get("experiment_name", "r-jepa")
    run_name = experiment_cfg.get("run_name")
    tags = dict(experiment_cfg.get("tags", {}))

    tracker = ExperimentTracker(
        enabled=True,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri or None,
        run_name=run_name,
        tags=tags,
    )
    if tracker.enabled:
        print(f"  Experiment tracking: MLflow → {experiment_name}")
        if tracking_uri:
            print(f"  Tracking URI: {tracking_uri}")
    else:
        print("  Experiment tracking: disabled (MLflow not available)")
    return tracker


def main() -> None:
    args = parse_args()
    config = override_config(load_config(args.config), args)
    device = get_device(args.device)

    # ── Experiment tracking (MLflow) ────────────────────────────
    experiment_tracker = _build_experiment_tracker(config, args)

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
    dc = config["data"]
    tickers_val = dc["tickers"]
    # If a single ticker was passed as --ticker, it's already a list;
    # otherwise the YAML may contain a list — both are fine for tuple().
    if isinstance(tickers_val, str):
        tickers_val = [tickers_val]
    data_cfg = StockDataConfig(
        tickers=tuple(tickers_val),
        start_date=dc["start_date"],
        end_date=dc["end_date"],
        sequence_length=dc["sequence_length"],
        prediction_horizon=dc["prediction_horizon"],
        features=tuple(dc["features"]),
        normalize=dc.get("normalize", True),
        train_split=dc.get("train_split", 0.8),
        val_split=dc.get("val_split", 0.1),
        sampling_interval=dc.get("sampling_interval"),
        use_high_freq_features=dc.get("use_high_freq_features", False),
        conv_downsample=config.get("model", {}).get("r_jepa", {}).get("conv_downsample", False),
        use_transformer_encoder=config.get("model", {}).get("r_jepa", {}).get("use_transformer", False),
        aug_noise_std=dc.get("aug_noise_std", 0.0),
        aug_magnitude_sigma=dc.get("aug_magnitude_sigma", 0.0),
        aug_time_warp_sigma=dc.get("aug_time_warp_sigma", 0.0),
        aug_window_slice_ratio=dc.get("aug_window_slice_ratio", 0.0),
    )
    loader = StockDataLoader(data_cfg)
    data = loader.preprocess()
    n_features = data.shape[1]
    print(f"  Total data shape: {data.shape}")
    print(f"  Feature names: {loader.get_feature_names()}")
    # Print active augmentations
    aug_parts = []
    if data_cfg.aug_noise_std > 0:
        aug_parts.append(f"noise(σ={data_cfg.aug_noise_std})")
    if data_cfg.aug_magnitude_sigma > 0:
        aug_parts.append(f"magnitude(σ={data_cfg.aug_magnitude_sigma})")
    if data_cfg.aug_time_warp_sigma > 0:
        aug_parts.append(f"time_warp(σ={data_cfg.aug_time_warp_sigma})")
    if data_cfg.aug_window_slice_ratio > 0:
        aug_parts.append(f"window_slice(r={data_cfg.aug_window_slice_ratio})")
    if aug_parts:
        print(f"  Augmentation (context-only): {', '.join(aug_parts)}")

    # Print curriculum learning status
    curriculum_cfg = config.get("training", {}).get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        init_h = curriculum_cfg.get("initial_horizon", 1)
        warmup = curriculum_cfg.get("warmup_epochs", 10)
        step = curriculum_cfg.get("step_epochs", 5)
        final_h = dc.get("prediction_horizon", 5)
        print(f"  Curriculum learning: H={init_h} → {final_h} "
              f"(warmup={warmup}, step={step})")
    print()

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

    # ── Step 4: Walk-Forward Validation (alternative to training) ──
    if args.walk_forward:
        print(f"Step 4a: Walk-forward validation ({args.wf_splits} folds)...")
        walk_config = WalkForwardConfig(
            n_splits=args.wf_splits,
            min_train_fraction=args.wf_min_train,
            batch_size=config["data"]["batch_size"],
            num_workers=2,
            verbose=True,
            early_stopping_patience=config["training"].get("patience", 10),
        )

        # Pass experiment tracker to each fold via trainer_builder
        def _wf_trainer_builder(m, train_l, val_l):
            return RJEPATrainer(
                m, train_l, val_l, config, device=device,
                experiment_tracker=experiment_tracker,
            )

        validator = WalkForwardValidator(
            data=data,
            walk_config=walk_config,
            model_config=mc,
            data_config=data_cfg,
            training_config=config["training"],
            full_config=config,
            device=device,
            trainer_builder=_wf_trainer_builder,
        )
        results = validator.run()

        # Save per-fold checkpoints for the best fold
        best_fold = min(results, key=lambda r: r.best_val_loss)
        print(f"\nBest fold: #{best_fold.fold + 1} "
              f"(val loss: {best_fold.best_val_loss:.6f})")

        summary = validator.summary()
        summary_path = os.path.join(
            config["training"]["save_dir"], "walk_forward_summary.json"
        )
        os.makedirs(config["training"]["save_dir"], exist_ok=True)
        import json
        (Path(summary_path)).write_text(json.dumps(summary, indent=2))
        print(f"  Summary saved to: {summary_path}")
        print()

        return  # Skip regular training

    # ── Step 4 (alt): Regular Training ──────────────────────────
    print("Step 4: Training...")
    trainer = RJEPATrainer(
        model, train_loader, val_loader, config, device=device,
        experiment_tracker=experiment_tracker,
    )
    # Persist StandardScaler parameters in every checkpoint
    if config["data"].get("normalize", True) and hasattr(loader, "_scaler"):
        trainer.set_scaler(loader._scaler.mean_, loader._scaler.scale_)  # type: ignore[union-attr]
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
