"""
Tests for JEPA loss functions.

Covers:
- ``variance_regularization`` — prevents collapse by enforcing std ≥ epsilon
- ``covariance_regularization`` — decorrelates latent dimensions
- ``JEPALoss`` — full loss combining prediction, variance, covariance
"""

from __future__ import annotations

import torch
from torch import Tensor

from training.loss import (
    JEPALoss,
    covariance_regularization,
    variance_regularization,
)


# ── variance_regularization ──────────────────────────────────────────────


def test_variance_regularization_high_variance() -> None:
    """When std >> epsilon, loss should be near zero (relu(clamp - std) = 0)."""
    z = torch.randn(128, 64) * 10.0  # high variance
    loss = variance_regularization(z, epsilon=0.001)
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item():.6f}"


def test_variance_regularization_low_variance() -> None:
    """When std << epsilon, loss should be positive (encourages higher variance)."""
    z = torch.ones(128, 64) * 0.001  # near-constant → std ≈ 0
    loss = variance_regularization(z, epsilon=1.0)
    # std ≈ 0, so relu(1.0 - 0) = 1.0
    assert loss.item() > 0.5, f"Expected positive loss, got {loss.item():.6f}"


def test_variance_regularization_batch_size_one() -> None:
    """Single sample should return zero loss (no batch variance possible)."""
    z = torch.randn(1, 64)
    loss = variance_regularization(z, epsilon=0.5)
    assert loss.item() == 0.0, (
        f"Expected 0.0 for batch_size=1, got {loss.item():.6f}"
    )


def test_variance_regularization_different_epsilon() -> None:
    """Larger epsilon should produce larger loss for the same input std."""
    z = torch.randn(128, 64) * 0.3
    loss_small = variance_regularization(z, epsilon=0.1)
    loss_large = variance_regularization(z, epsilon=1.0)
    assert loss_large > loss_small, (
        f"Expected loss(ε=1.0)={loss_large:.6f} > loss(ε=0.1)={loss_small:.6f}"
    )


# ── covariance_regularization ────────────────────────────────────────────


def test_covariance_regularization_diagonal() -> None:
    """Perfectly decorrelated batch → near-zero loss."""
    z = torch.randn(128, 64)  # independent dims → near-diagonal cov
    loss = covariance_regularization(z)
    # Off-diagonal elements of cov for random data should be small
    assert loss.item() < 0.5, f"Expected small loss, got {loss.item():.6f}"


def test_covariance_regularization_collinear() -> None:
    """Collinear dimensions → high off-diagonal → large loss."""
    z = torch.randn(128, 2)
    # Make second column = first column + noise → high correlation
    z[:, 1] = z[:, 0] + torch.randn(128) * 0.01
    loss = covariance_regularization(z)
    assert loss.item() > 0.01, f"Expected non-trivial loss, got {loss.item():.6f}"


def test_covariance_regularization_batch_size_one() -> None:
    """Single sample → zero loss."""
    z = torch.randn(1, 64)
    loss = covariance_regularization(z)
    assert loss.item() == 0.0, f"Expected 0.0 for batch_size=1, got {loss.item():.6f}"


# ── JEPALoss ─────────────────────────────────────────────────────────────


class TestJEPALoss:
    """Tests for the full JEPA loss function."""

    def test_loss_output_structure(self) -> None:
        """Forward returns dict with expected keys."""
        B, T, L = 4, 5, 8
        predicted = torch.randn(B, T, L)
        target_latent = torch.randn(B, L)
        criterion = JEPALoss()

        result = criterion(
            predicted_latents=predicted,
            target_latent=target_latent,
        )

        expected_keys = {"loss", "prediction_loss", "variance_loss", "covariance_loss"}
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_loss_positive(self) -> None:
        """Loss is strictly positive for random inputs."""
        B, T, L = 4, 5, 8
        predicted = torch.randn(B, T, L)
        target_latent = torch.randn(B, L)
        criterion = JEPALoss()

        result = criterion(predicted_latents=predicted, target_latent=target_latent)
        assert result["loss"].item() > 0, (
            f"Expected positive loss, got {result['loss'].item():.6f}"
        )

    def test_perfect_prediction_zero_pred_loss(self) -> None:
        """When prediction exactly matches target, prediction loss should be small."""
        B, L = 4, 8
        target_latent = torch.randn(B, L)
        # Perfect prediction: predicted_latents mean = target_latent
        predicted = target_latent.unsqueeze(1).expand(-1, 5, -1)
        criterion = JEPALoss(variance_weight=0.0, covariance_weight=0.0)

        result = criterion(
            predicted_latents=predicted,
            target_latent=target_latent,
        )
        assert result["prediction_loss"].item() < 1e-6, (
            f"Expected near-zero prediction loss, got {result['prediction_loss'].item():.6f}"
        )

    def test_reconstruction_loss_included(self) -> None:
        """When decoder outputs are passed, reconstruction loss is added."""
        B, T, F = 4, 5, 3
        predicted = torch.randn(B, T, 8)
        target_latent = torch.randn(B, 8)
        decoded = torch.randn(B, T, F)
        target = torch.randn(B, T, F)
        criterion = JEPALoss(reconstruction_weight=0.5)

        result = criterion(
            predicted_latents=predicted,
            target_latent=target_latent,
            decoded_predictions=decoded,
            target=target,
        )
        assert "reconstruction_loss" in result, "Missing reconstruction_loss key"
        assert result["reconstruction_loss"].item() > 0

    def test_reconstruction_loss_skipped_without_target(self) -> None:
        """Without decoded/target args, reconstruction loss is absent."""
        B, T, L = 4, 5, 8
        predicted = torch.randn(B, T, L)
        target_latent = torch.randn(B, L)
        criterion = JEPALoss()

        result = criterion(
            predicted_latents=predicted,
            target_latent=target_latent,
        )
        assert "reconstruction_loss" not in result

    def test_regularization_weights_affect_loss(self) -> None:
        """Higher variance_weight should increase the variance loss contribution."""
        B, T, L = 4, 5, 8
        predicted = torch.randn(B, T, L)
        # Near-zero variance latent — std ≈ 0, so var_loss > 0
        target_latent = torch.ones(B, L) * 0.001

        # Use epsilon=1.0 so that near-zero std produces clear var_loss
        criterion_high = JEPALoss(
            variance_weight=10.0, variance_epsilon=1.0, covariance_weight=0.0,
        )
        criterion_low = JEPALoss(
            variance_weight=0.0, variance_epsilon=1.0, covariance_weight=0.0,
        )

        result_high = criterion_high(
            predicted_latents=predicted, target_latent=target_latent
        )
        result_low = criterion_low(
            predicted_latents=predicted, target_latent=target_latent
        )
        assert result_high["loss"].item() > result_low["loss"].item(), (
            f"Expected high-var-weight loss ({result_high['loss'].item():.6f}) > "
            f"low-var-weight loss ({result_low['loss'].item():.6f})"
        )

    def test_differentiable(self) -> None:
        """Loss is differentiable — backward pass succeeds."""
        B, T, L = 4, 5, 8
        predicted = torch.randn(B, T, L, requires_grad=True)
        target_latent = torch.randn(B, L)
        criterion = JEPALoss()

        result = criterion(predicted_latents=predicted, target_latent=target_latent)
        result["loss"].backward()
        assert predicted.grad is not None, "Gradient should be populated"
        assert predicted.grad.abs().sum().item() > 0, "Gradient should be non-zero"
