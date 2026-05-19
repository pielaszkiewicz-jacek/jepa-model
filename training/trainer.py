"""
R-JEPA Trainer.
Handles the training loop, validation, checkpointing, and logging.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.r_jepa import RJEPA
from training.loss import JEPALoss


class RJEPATrainer:
    """
    Trainer for R-JEPA model with self-supervised learning.

    Features:
    - EMA momentum encoder updates
    - Automatic Mixed Precision (AMP)
    - Gradient clipping
    - Early stopping
    - Checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: RJEPA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Mapping[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        training_config = config.get("training", {})
        model_config = config.get("model", {}).get("r_jepa", {})

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

        # Loss
        self.criterion = JEPALoss(
            variance_weight=0.5,
            covariance_weight=0.1,
            predictor_epsilon=model_config.get("predictor_epsilon", 0.001),
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.get("num_epochs", 100),
            eta_min=1e-6,
        )

        # AMP
        self.use_amp = training_config.get("use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Hyper-parameters
        self.num_epochs = training_config.get("num_epochs", 100)
        self.patience = training_config.get("patience", 10)
        self.gradient_clip_val = training_config.get("gradient_clip_val", 1.0)
        self.log_interval = training_config.get("log_interval", 10)
        self.save_dir = Path(training_config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Gradient accumulation — effective batch = batch_size × accumulation_steps
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 1
        )
        if self.gradient_accumulation_steps > 1:
            print(f"  Gradient accumulation: {self.gradient_accumulation_steps} steps")
            print(f"  Effective batch size: "
                  f"{self.train_loader.batch_size * self.gradient_accumulation_steps}")

        # State
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.learning_rates: list[float] = []

    # ── Public API ──────────────────────────────────────────────

    def train(self) -> RJEPA:
        """Run the full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}\n")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1

            train_metrics = self._train_epoch()
            self.train_losses.append(train_metrics["loss"])
            self.learning_rates.append(self.scheduler.get_last_lr()[0])

            val_metrics = self._validate()
            val_loss = val_metrics["val_loss"]
            self.val_losses.append(val_loss)

            self.scheduler.step()

            if epoch % self.log_interval == 0 or epoch == self.num_epochs - 1:
                print(
                    f"Epoch {self.current_epoch:3d}/{self.num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            if epoch % 10 == 0:
                self._save_checkpoint()

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                break

        self._save_checkpoint()
        self._save_metrics()

        print(f"\nTraining completed!")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        print(f"  Checkpoints saved to: {self.save_dir}")

        # Load best checkpoint
        best_path = self.save_dir / "checkpoint_best.pt"
        if best_path.exists():
            self._load_checkpoint(str(best_path))

        return self.model

    # ── Internal: training steps ────────────────────────────────

    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch and return average losses.

        Supports gradient accumulation for high-frequency data where
        the number of batches per epoch can be very large.
        """
        self.model.train()

        total_loss = 0.0
        total_pred = 0.0
        total_var = 0.0
        total_cov = 0.0
        total_recon = 0.0
        n_batches = 0

        accumulation_steps = self.gradient_accumulation_steps
        optim_steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (context, target) in enumerate(pbar):
            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                output = self.model(context=context, target=target)
                loss_dict = self.criterion(
                    predicted_latents=output["predicted_latents"],
                    target_latent=output["target_latent"],
                    context_latent=output["context_latent"],
                    decoded_predictions=output["decoded_predictions"],
                    target=target,
                )
                # Scale loss for gradient accumulation
                loss = loss_dict["loss"] / accumulation_steps

            self.scaler.scale(loss).backward()

            # Accumulate gradients for N steps, then step optimizer
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.model.update_momentum_encoder()
                optim_steps += 1

            total_loss += loss_dict["loss"].item()
            total_pred += loss_dict["prediction_loss"].item()
            total_var += loss_dict["variance_loss"].item()
            total_cov += loss_dict["covariance_loss"].item()
            if "reconstruction_loss" in loss_dict:
                total_recon += loss_dict["reconstruction_loss"].item()
            n_batches += 1

            pbar.set_postfix(loss=loss_dict["loss"].item())

        # Ensure gradients from last partial accumulation are applied
        if n_batches % accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.model.update_momentum_encoder()
            optim_steps += 1

        n = max(n_batches, 1)
        result: dict[str, float] = {
            "loss": total_loss / n,
            "prediction_loss": total_pred / n,
            "variance_loss": total_var / n,
            "covariance_loss": total_cov / n,
        }
        if total_recon > 0:
            result["reconstruction_loss"] = total_recon / n
        return result

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation and return average validation loss."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        n_batches = 0

        for context, target in self.val_loader:
            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(context=context, target=target)
            loss_dict = self.criterion(
                predicted_latents=output["predicted_latents"],
                target_latent=output["target_latent"],
                context_latent=output["context_latent"],
                decoded_predictions=output["decoded_predictions"],
                target=target,
            )

            total_loss += loss_dict["loss"].item()
            if "reconstruction_loss" in loss_dict:
                total_recon += loss_dict["reconstruction_loss"].item()
            n_batches += 1

        result: dict[str, float] = {"val_loss": total_loss / max(n_batches, 1)}
        if total_recon > 0:
            result["val_reconstruction_loss"] = total_recon / max(n_batches, 1)
        return result

    # ── Checkpointing ────────────────────────────────────────────

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": dict(self.config),
        }

        torch.save(ckpt, self.save_dir / "checkpoint_latest.pt")

        if is_best:
            torch.save(ckpt, self.save_dir / "checkpoint_best.pt")
            print(f"  New best model saved! Val loss: {self.best_val_loss:.6f}")

    def _load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]
        self.current_epoch = ckpt["epoch"]
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _save_metrics(self) -> None:
        """Save training metrics to JSON."""
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "config": dict(self.config),
        }
        (self.save_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
