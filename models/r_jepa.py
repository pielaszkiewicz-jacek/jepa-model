"""
R-JEPA (Recurrent Joint Embedding Predictive Architecture) for Stock Prediction.

R-JEPA combines:
1. JEPA (Joint Embedding Predictive Architecture): Predict in latent space
   rather than in observation space
2. Recurrent processing: GRU/LSTM for temporal dynamics
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models.encoder import TimeSeriesEncoder, MomentumEncoder
from models.predictor import RecurrentPredictor
from models.decoder import StockDecoder


class RJEPA(nn.Module):
    """
    R-JEPA Model for Stock Price Prediction.

    Architecture:
    - Online Encoder: Encodes context windows to latent space
    - Momentum Encoder: Encodes target windows (EMA-updated, no gradients)
    - Recurrent Predictor: Predicts future latents from context latents
    - Decoder: (Optional) Decodes latents back to stock price space
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dim: int = 128,
        encoder_num_layers: int = 2,
        latent_dim: int = 64,
        predictor_hidden_dim: int = 128,
        predictor_num_layers: int = 2,
        predictor_dropout: float = 0.1,
        decoder_hidden_dim: int = 64,
        decoder_output_dim: int | None = None,
        momentum_tau: float = 0.996,
        conv_downsample: bool = False,
        use_transformer: bool = False,
        nhead: int = 8,
    ) -> None:
        super().__init__()

        # ── Parameter validation ─────────────────────────────────
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if encoder_hidden_dim < 1:
            raise ValueError(f"encoder_hidden_dim must be >= 1, got {encoder_hidden_dim}")
        if encoder_num_layers < 1:
            raise ValueError(f"encoder_num_layers must be >= 1, got {encoder_num_layers}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if predictor_hidden_dim < 1:
            raise ValueError(f"predictor_hidden_dim must be >= 1, got {predictor_hidden_dim}")
        if predictor_num_layers < 1:
            raise ValueError(f"predictor_num_layers must be >= 1, got {predictor_num_layers}")
        if not 0 <= predictor_dropout < 1:
            raise ValueError(f"predictor_dropout must be in [0, 1), got {predictor_dropout}")
        if decoder_hidden_dim < 1:
            raise ValueError(f"decoder_hidden_dim must be >= 1, got {decoder_hidden_dim}")
        if not 0 < momentum_tau < 1:
            raise ValueError(f"momentum_tau must be in (0, 1), got {momentum_tau}")
        if nhead < 1:
            raise ValueError(f"nhead must be >= 1, got {nhead}")

        self.latent_dim = latent_dim
        decoder_output_dim = decoder_output_dim or input_dim

        self.online_encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            latent_dim=latent_dim,
            conv_downsample=conv_downsample,
            use_transformer=use_transformer,
            nhead=nhead,
        )

        self.momentum_encoder = MomentumEncoder(
            encoder=self.online_encoder,
            tau=momentum_tau,
        )

        self.predictor = RecurrentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=predictor_dropout,
        )

        self.decoder = StockDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=decoder_output_dim,
        )

    def _encode_context(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode context sequence and extract the final latent state.

        Shared helper used by both :meth:`forward` and :meth:`predict`
        to avoid duplicating the encoder call.

        Args:
            context: Input tensor of shape ``[batch, seq_len, n_features]``.

        Returns:
            Tuple of ``(context_latent_seq, context_latent)`` where:
            - ``context_latent_seq`` — full encoder output sequence
              ``[batch, seq_len, latent_dim]``.
            - ``context_latent`` — last time-step latent
              ``[batch, latent_dim]``.
        """
        context_latent_seq = self.online_encoder(context, return_sequence=True)
        context_latent = context_latent_seq[:, -1, :]
        return context_latent_seq, context_latent

    def forward(
        self,
        context: Tensor,
        target: Tensor | None = None,
        prediction_horizon: int | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for both training and inference."""
        if prediction_horizon is None:
            if target is None:
                raise ValueError("Either target or prediction_horizon must be provided")
            prediction_horizon = target.shape[1]

        # Encode context
        context_latent_seq, context_latent = self._encode_context(context)

        # Predict future latents
        predicted_latents = self.predictor(
            context_latent=context_latent,
            prediction_horizon=prediction_horizon,
            context_latent_seq=context_latent_seq,
        )

        result: dict[str, Tensor] = {
            "predicted_latents": predicted_latents,
            "context_latent": context_latent,
            "context_latent_seq": context_latent_seq,
        }

        if target is not None:
            result["target_latent"] = self.momentum_encoder(target)
            result["target"] = target

        result["decoded_predictions"] = self.decoder.decode_sequence(predicted_latents)
        return result

    @torch.no_grad()
    def predict(
        self,
        context: Tensor,
        prediction_horizon: int,
        return_ensemble: bool = False,
        ensemble_size: int = 5,
        noise_scale: float = 0.05,
    ) -> dict[str, Tensor]:
        """Predict future stock prices with optional uncertainty estimation."""
        self.eval()

        if not return_ensemble:
            output = self.forward(context=context, prediction_horizon=prediction_horizon)
            return {
                "predictions": output["decoded_predictions"],
                "latents": output["predicted_latents"],
            }

        # Encode context (shared helper)
        context_latent_seq, context_latent = self._encode_context(context)

        ensemble_latents = self.predictor.predict_with_noise(
            context_latent=context_latent,
            prediction_horizon=prediction_horizon,
            noise_scale=noise_scale,
            num_samples=ensemble_size,
        )  # [B, ensemble_size, pred_horizon, latent_dim]

        B, N, T, L = ensemble_latents.shape
        decoded = self.decoder.decode_sequence(
            ensemble_latents.reshape(-1, T, L)
        ).reshape(B, N, T, -1)

        mean_pred = decoded.mean(dim=1)
        std_pred = decoded.std(dim=1)
        confidence = 1.0 / (1.0 + std_pred.mean(dim=-1))

        return {
            "predictions": mean_pred,
            "ensemble": decoded,
            "confidence": confidence,
            "upper_bound": mean_pred + 1.96 * std_pred,
            "lower_bound": mean_pred - 1.96 * std_pred,
        }

    def update_momentum_encoder(self) -> None:
        """Update momentum encoder weights via EMA."""
        self.momentum_encoder.update_momentum()
