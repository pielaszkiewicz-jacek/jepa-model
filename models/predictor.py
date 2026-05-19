"""
Recurrent Predictor for R-JEPA.
Predicts future latent representations from current context latents.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RecurrentPredictor(nn.Module):
    """
    Recurrent predictor that transforms encoded context latents
    into predicted future latents in the latent space.

    Uses a GRU with autoregressive decoding to iteratively predict
    future latent representations step by step.

    Architecture:
        context_latent [B, L]
            │
            ├── (optional) context_latent_seq [B, T, L] → GRU init
            │
            ├── context_aggregator: Linear(L→hidden) → LayerNorm → GELU
            │       (projects latent to GRU hidden dimension for init)
            │
            ├── GRU(hidden→hidden, num_layers) — autoregressive loop
            │
            └── output_projection: Linear(hidden→hidden)→LN→GELU→Dropout
                                   → Linear(hidden→L)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Parameter validation ─────────────────────────────────
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.latent_dim = latent_dim

        self.context_aggregator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        context_latent: Tensor,
        prediction_horizon: int,
        context_latent_seq: Tensor | None = None,
    ) -> Tensor:
        """
        Predict future latent representations.

        Args:
            context_latent: Encoded context [batch_size, latent_dim]
            prediction_horizon: Number of future steps to predict
            context_latent_seq: Optional full latent sequence
                                [batch_size, seq_len, latent_dim]

        Returns:
            Predicted future latents [batch_size, prediction_horizon, latent_dim]
        """
        # Initialise GRU hidden state
        if context_latent_seq is not None:
            # Process full context sequence → last hidden as initial state
            # GRU input:  [B, T, latent_dim]  (matches input_size=latent_dim)
            # GRU output: [num_layers, B, hidden_dim]
            _, hidden = self.gru(context_latent_seq)
        else:
            # Use context_aggregator to project latent_dim → hidden_dim
            # and use it as the initial GRU hidden state
            # agg:       [B, hidden_dim]
            # hidden:    [num_layers, B, hidden_dim]  (repeat across layers)
            agg = self.context_aggregator(context_latent)
            hidden = agg.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)

        # Autoregressive input starts as the context latent
        # GRU expects input_size=latent_dim throughout the loop
        current = context_latent.unsqueeze(1)  # [B, 1, latent_dim]
        predictions: list[Tensor] = []

        for _ in range(prediction_horizon):
            gru_out, hidden = self.gru(current, hidden)       # [B, 1, hidden]
            pred = self.output_projection(gru_out)            # [B, 1, latent_dim]
            predictions.append(pred)
            current = pred

        return torch.cat(predictions, dim=1)  # [B, pred_horizon, latent_dim]

    @torch.no_grad()
    def predict_with_noise(
        self,
        context_latent: Tensor,
        prediction_horizon: int,
        noise_scale: float = 0.05,
        num_samples: int = 5,
    ) -> Tensor:
        """
        Generate multiple prediction trajectories with noise injection.
        Useful for uncertainty estimation and ensemble predictions.

        Returns:
            Predictions [batch_size, num_samples, prediction_horizon, latent_dim]
        """
        batch_size = context_latent.shape[0]

        context_expanded = (
            context_latent.unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(batch_size * num_samples, -1)
        )

        # Initialise hidden state via context_aggregator (latent_dim → hidden_dim)
        agg = self.context_aggregator(context_expanded)  # [B*N, hidden_dim]
        hidden = agg.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # [num_layers, B*N, hidden_dim]

        current = context_expanded.unsqueeze(1)  # [B*N, 1, latent_dim]

        predictions: list[Tensor] = []
        for _ in range(prediction_horizon):
            noisy_input = current
            if noise_scale > 0:
                noisy_input = current + torch.randn_like(current) * noise_scale

            gru_out, hidden = self.gru(noisy_input, hidden)
            pred = self.output_projection(gru_out)
            predictions.append(pred)
            current = pred

        stacked = torch.stack(predictions, dim=1)  # [B*N, pred_horizon, latent_dim]
        return stacked.reshape(batch_size, num_samples, prediction_horizon, -1)
