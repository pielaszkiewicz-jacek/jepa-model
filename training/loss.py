"""
JEPA Loss Functions.

Combines prediction error (MSE in latent space) with variance and covariance
regularization to prevent representation collapse.

Hybrid mode: optionally adds supervised reconstruction loss (MSE in observation
space) to train the decoder jointly with the JEPA objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def variance_regularization(z: Tensor, epsilon: float = 0.001) -> Tensor:
    """
    Variance regularization to prevent collapse.
    Encourages each latent dimension to have std >= epsilon.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-10)
    return F.relu(epsilon - std).mean()


def covariance_regularization(z: Tensor) -> Tensor:
    """
    Covariance regularization to decorrelate latent dimensions.
    Penalizes off-diagonal elements of the covariance matrix.
    """
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z.shape[0] - 1)

    # Sum of squared off-diagonal elements
    off_diag = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[1] + 1)[:, 1:]
    return off_diag.pow(2).sum() / cov.shape[0]


class JEPALoss(nn.Module):
    """
    JEPA Loss combining:
    1. Prediction error (MSE in latent space)
    2. Variance regularization (prevents collapse)
    3. Covariance regularization (decorrelates dimensions)
    4. [Hybrid] Supervised reconstruction loss (MSE in observation space)

    The reconstruction loss trains the decoder jointly with JEPA,
    enabling the model to produce meaningful price predictions.
    """

    def __init__(
        self,
        variance_weight: float = 0.5,
        covariance_weight: float = 0.1,
        predictor_epsilon: float = 0.001,
        reconstruction_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.predictor_epsilon = predictor_epsilon
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        predicted_latents: Tensor,
        target_latent: Tensor,
        context_latent: Tensor | None = None,
        decoded_predictions: Tensor | None = None,
        target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute JEPA loss with optional hybrid reconstruction loss.

        Args:
            predicted_latents: [batch_size, pred_horizon, latent_dim]
            target_latent: [batch_size, latent_dim]
            context_latent: Optional [batch_size, latent_dim]
            decoded_predictions: Optional [batch_size, pred_horizon, n_features]
                Decoder output — if provided, reconstruction loss is computed.
            target: Optional [batch_size, pred_horizon, n_features]
                Ground-truth future prices for reconstruction loss.

        Returns:
            Dictionary with loss components
        """
        # ── JEPA prediction loss (latent space) ──────────────────
        pred_mean = predicted_latents.mean(dim=1)          # [B, latent_dim]
        pred_mse = F.mse_loss(pred_mean, target_latent)

        target_expanded = target_latent.unsqueeze(1).expand_as(predicted_latents)
        step_mse = F.mse_loss(predicted_latents, target_expanded)

        total_prediction_loss = 0.5 * pred_mse + 0.5 * step_mse

        # ── Variance regularization ──────────────────────────────
        var_loss = 0.5 * (
            variance_regularization(pred_mean, self.predictor_epsilon)
            + variance_regularization(target_latent, self.predictor_epsilon)
        )

        # ── Covariance regularization ────────────────────────────
        cov_loss = 0.5 * (
            covariance_regularization(pred_mean)
            + covariance_regularization(target_latent)
        )

        total_loss = (
            total_prediction_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )

        result: dict[str, Tensor] = {
            "loss": total_loss,
            "prediction_loss": total_prediction_loss.detach(),
            "variance_loss": var_loss.detach(),
            "covariance_loss": cov_loss.detach(),
        }

        # ── Hybrid: supervised reconstruction loss (price space) ─
        if decoded_predictions is not None and target is not None:
            recon_loss = F.mse_loss(decoded_predictions, target)
            total_loss = total_loss + self.reconstruction_weight * recon_loss

            result["loss"] = total_loss
            result["reconstruction_loss"] = recon_loss.detach()

        return result
