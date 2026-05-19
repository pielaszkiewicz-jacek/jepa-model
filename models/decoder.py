"""
Stock Decoder for R-JEPA.
Decodes latent representations back to stock price space.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StockDecoder(nn.Module):
    """
    Decoder that transforms latent representations back to stock price space.
    Useful for converting predictions back to stock prices and visualization.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        # ── Parameter validation ─────────────────────────────────
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = []
        current_dim = latent_dim

        for i in range(num_layers - 1):
            next_dim = hidden_dim if i < num_layers - 2 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 2:
                layers.append(nn.LayerNorm(next_dim))
                layers.append(nn.GELU())
            current_dim = next_dim

        # Only add the final projection when the last loop iteration didn't
        # already project to output_dim, eliminating the degenerate
        # Linear(output_dim → output_dim) layer.
        if current_dim != output_dim:
            layers.append(nn.Linear(current_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Decode latent representation to stock price space.

        Args:
            latent: [batch_size, latent_dim] or [batch_size, seq_len, latent_dim]

        Returns:
            Decoded output with same batch/seq dimensions
        """
        return self.decoder(latent)

    def decode_sequence(self, latent_seq: Tensor) -> Tensor:
        """
        Decode a sequence of latent representations.

        Args:
            latent_seq: [batch_size, seq_len, latent_dim]

        Returns:
            Decoded sequence [batch_size, seq_len, output_dim]
        """
        *dims, L = latent_seq.shape
        flat = latent_seq.view(-1, L)
        decoded = self.decoder(flat)
        return decoded.view(*dims, -1)
