"""
Time Series Encoder for R-JEPA.
Transforms input stock time series into latent representations.

Supports two modes:
- **Standard** (BiLSTM): for daily / short sequences (seq_len ≤ 120)
- **High-frequency** (Conv1D downsample + Transformer): for long sequences
  (seq_len ≥ 600), suitable for 1s / 100ms resolution data.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import Tensor, nn


class TimeSeriesEncoder(nn.Module):
    """
    Encoder module that transforms raw time series data into latent representations.

    Architecture depends on ``conv_downsample`` and ``use_transformer``:

    === Standard mode (lstm) ===
        Conv1D(k=3, pad=1) → BN → GELU → Conv1D → BN → GELU → BiLSTM → Projection

    === High-frequency mode (conv_downsample + transformer) ===
        Conv1D(k=32, s=4) → BN → GELU → Conv1D(k=16, s=2) → BN → GELU
        → TransformerEncoder (nhead, num_layers) → Projection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        conv_downsample: bool = False,
        use_transformer: bool = False,
        nhead: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        self.conv_downsample = conv_downsample

        # ── Conv1D front-end ──────────────────────────────────────
        if conv_downsample:
            # Multi-scale downsampling for long sequences
            # e.g. T=600 → 150 → 75 → 75 tokens
            self.conv_frontend = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 2,
                          kernel_size=32, stride=4, padding=16),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim,
                          kernel_size=16, stride=2, padding=8),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim,
                          kernel_size=8, stride=1, padding=4),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            proj_input_dim = hidden_dim
        else:
            # Standard shallow convolutions (no downsampling)
            self.conv_frontend = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 2,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            proj_input_dim = hidden_dim

        # ── Temporal model ─────────────────────────────────────────
        if use_transformer:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer

            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_model = TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            proj_input_dim = hidden_dim
        else:
            self.temporal_model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )
            proj_input_dim = hidden_dim * 2

        # ── Projection to latent space ─────────────────────────────
        if use_transformer or conv_downsample:
            # Single direction → no *2 needed
            self.projection = nn.Sequential(
                nn.Linear(proj_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: Tensor, return_sequence: bool = False
    ) -> Tensor:
        """
        Encode input time series to latent representation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            return_sequence: If True, return full latent sequence

        Returns:
            If return_sequence: [batch_size, seq_len_out, latent_dim]
            Else: [batch_size, latent_dim]
            Note: ``seq_len_out`` may be shorter than input when
            ``conv_downsample=True`` (downsampled by stride factors).
        """
        # Conv1d expects [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)                     # [B, input_dim, seq_len]
        conv_out = self.conv_frontend(x_conv)           # [B, hidden_dim, seq_len']
        seq_len_out = conv_out.shape[-1]
        conv_out = conv_out.transpose(1, 2)             # [B, seq_len', hidden_dim]

        # ── Temporal model ────────────────────────────────────────
        if self.use_transformer:
            # TransformerEncoder: [B, seq_len', hidden_dim]
            encoded = self.temporal_model(conv_out)

            if return_sequence:
                return self.projection(encoded)         # [B, seq_len', latent_dim]

            # Global mean pooling over sequence
            pooled = encoded.mean(dim=1)                 # [B, hidden_dim]
            return self.projection(pooled)               # [B, latent_dim]

        else:
            # BiLSTM: [B, seq_len', hidden_dim*2]
            lstm_out, (hidden, _) = self.temporal_model(conv_out)

            if return_sequence:
                return self.projection(lstm_out)         # [B, seq_len', latent_dim]

            # Concatenate forward + backward from last layer
            last_forward, last_backward = hidden[-2], hidden[-1]
            combined = torch.cat([last_forward, last_backward], dim=-1)
            return self.projection(combined)             # [B, latent_dim]


class MomentumEncoder(nn.Module):
    """
    Momentum encoder for R-JEPA.
    Maintains a slowly-updated copy of the online encoder for stable training.
    Parameters are updated via Exponential Moving Average (EMA).
    """

    def __init__(self, encoder: TimeSeriesEncoder, tau: float = 0.996) -> None:
        super().__init__()

        self.encoder = encoder
        self.tau = tau

        # Infer dimensions from the online encoder
        first_conv = encoder.conv_frontend[0]
        last_proj = encoder.projection[-2]  # Linear(latent_dim) before LayerNorm

        # Infer number of layers from whichever temporal model is active
        if hasattr(encoder.temporal_model, "num_layers"):
            inferred_layers = encoder.temporal_model.num_layers  # LSTM
        elif hasattr(encoder.temporal_model, "layers"):
            inferred_layers = len(encoder.temporal_model.layers)  # Transformer
        else:
            inferred_layers = 2  # fallback

        # Infer number of attention heads if Transformer is active
        if (
            encoder.use_transformer
            and hasattr(encoder.temporal_model, "layers")
            and len(encoder.temporal_model.layers) > 0
        ):
            inferred_nhead = (
                encoder.temporal_model.layers[0].self_attn.num_heads
            )
        else:
            inferred_nhead = 8

        self.momentum_encoder = TimeSeriesEncoder(
            input_dim=first_conv.in_channels,           # type: ignore[union-attr]
            hidden_dim=encoder.hidden_dim,
            num_layers=inferred_layers,
            latent_dim=last_proj.out_features,          # type: ignore[union-attr]
            dropout=0.1,
            conv_downsample=encoder.conv_downsample,
            use_transformer=encoder.use_transformer,
            nhead=inferred_nhead,
        )
        self.momentum_encoder.load_state_dict(encoder.state_dict())

        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_momentum(self) -> None:
        """Update momentum encoder weights via EMA."""
        for online_param, momentum_param in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            momentum_param.data.lerp_(online_param.data, 1 - self.tau)

    def forward(self, x: Tensor, return_sequence: bool = False) -> Tensor:
        """Forward pass through momentum encoder (no gradients)."""
        with torch.no_grad():
            return self.momentum_encoder(x, return_sequence)

    @torch.no_grad()
    def forward_no_grad(self, x: Tensor, return_sequence: bool = False) -> Tensor:
        """Explicit no-grad forward pass."""
        return self.momentum_encoder(x, return_sequence)
