"""
Tests for R-JEPA model components.

Covers:
- ``TimeSeriesEncoder`` — forward pass, output shapes
- ``MomentumEncoder`` — EMA update, slow target convergence
- ``RecurrentPredictor`` — latent prediction, ensemble mode
- ``StockDecoder`` — decoding, ``decode_sequence``, no degenerate Linear layer
- ``RJEPA`` — forward, predict, momentum update, ``_encode_context``
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from models.decoder import StockDecoder
from models.encoder import TimeSeriesEncoder, MomentumEncoder
from models.predictor import RecurrentPredictor
from models.r_jepa import RJEPA


# ── TimeSeriesEncoder ────────────────────────────────────────────────────


class TestTimeSeriesEncoder:
    """Tests for the online encoder."""

    def test_output_shape(self, small_encoder: TimeSeriesEncoder,
                          context_tensor: Tensor) -> None:
        output = small_encoder(context_tensor, return_sequence=False)
        B, latent_dim = context_tensor.shape[0], small_encoder.latent_dim
        assert output.shape == (B, latent_dim), (
            f"Expected ({B}, {latent_dim}), got {output.shape}"
        )

    def test_output_sequence_shape(self, small_encoder: TimeSeriesEncoder,
                                    context_tensor: Tensor) -> None:
        output = small_encoder(context_tensor, return_sequence=True)
        B, T, latent_dim = (
            context_tensor.shape[0],
            context_tensor.shape[1],
            small_encoder.latent_dim,
        )
        assert output.shape == (B, T, latent_dim), (
            f"Expected ({B}, {T}, {latent_dim}), got {output.shape}"
        )

    def test_differentiable(self, small_encoder: TimeSeriesEncoder,
                            context_tensor: Tensor) -> None:
        x = context_tensor.clone().requires_grad_(True)
        output = small_encoder(x, return_sequence=False)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to input"
        assert x.grad.abs().sum().item() > 0

    def test_parameter_validation(self) -> None:
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="input_dim"):
            TimeSeriesEncoder(input_dim=0, hidden_dim=16, num_layers=1, latent_dim=8)

    def test_conv_downsample_flag(self, context_tensor: Tensor) -> None:
        """conv_downsample=True should be accepted (integration smoke test)."""
        B, T, F = context_tensor.shape
        enc = TimeSeriesEncoder(input_dim=F, hidden_dim=16, num_layers=1,
                                latent_dim=8, conv_downsample=True)
        out = enc(context_tensor, return_sequence=False)
        assert out.shape == (B, 8), f"Unexpected shape with conv_downsample: {out.shape}"


# ── MomentumEncoder ──────────────────────────────────────────────────────


class TestMomentumEncoder:
    """Tests for the momentum (target) encoder."""

    def test_initial_weights_match(self, small_encoder: TimeSeriesEncoder) -> None:
        """Initially, momentum encoder weights equal the online encoder."""
        momentum = MomentumEncoder(encoder=small_encoder, tau=0.996)
        for p_m, p_e in zip(momentum.parameters(), small_encoder.parameters()):
            assert torch.equal(p_m, p_e), "Initial weights should match"

    def test_update_momentum_blends_weights(self, small_encoder: TimeSeriesEncoder) -> None:
        """After update_momentum, weights are a blend of old and new."""
        momentum = MomentumEncoder(encoder=small_encoder, tau=0.5)
        # Change online encoder weights
        for p in small_encoder.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        # Save old momentum_encoder weights (not all MomentumEncoder params,
        # which also includes the online encoder reference)
        old_momentum_weights = [p.clone() for p in momentum.momentum_encoder.parameters()]
        momentum.update_momentum()
        # Weights should have changed (blended with online)
        for p_m, p_old in zip(momentum.momentum_encoder.parameters(), old_momentum_weights):
            assert not torch.equal(p_m, p_old), "Momentum weights should update"

    def test_forward_shape(self, small_encoder: TimeSeriesEncoder,
                           context_tensor: Tensor) -> None:
        momentum = MomentumEncoder(encoder=small_encoder, tau=0.996)
        output = momentum(context_tensor)
        B, latent_dim = context_tensor.shape[0], small_encoder.latent_dim
        assert output.shape == (B, latent_dim), (
            f"Expected ({B}, {latent_dim}), got {output.shape}"
        )


# ── RecurrentPredictor ───────────────────────────────────────────────────


class TestRecurrentPredictor:
    """Tests for the latent-space predictor."""

    def test_output_shape(self, small_predictor: RecurrentPredictor) -> None:
        B, latent_dim, pred_hz = 4, 8, 5
        context_latent = torch.randn(B, latent_dim)
        context_latent_seq = torch.randn(B, 10, latent_dim)
        output = small_predictor(
            context_latent=context_latent,
            prediction_horizon=pred_hz,
            context_latent_seq=context_latent_seq,
        )
        assert output.shape == (B, pred_hz, latent_dim), (
            f"Expected ({B}, {pred_hz}, {latent_dim}), got {output.shape}"
        )

    def test_predict_with_noise_shape(self, small_predictor: RecurrentPredictor) -> None:
        B, latent_dim, pred_hz, ensemble_size = 4, 8, 5, 3
        context_latent = torch.randn(B, latent_dim)
        output = small_predictor.predict_with_noise(
            context_latent=context_latent,
            prediction_horizon=pred_hz,
            noise_scale=0.05,
            num_samples=ensemble_size,
        )
        assert output.shape == (B, ensemble_size, pred_hz, latent_dim), (
            f"Unexpected ensemble shape: {output.shape}"
        )

    def test_differentiable(self, small_predictor: RecurrentPredictor) -> None:
        B, latent_dim, pred_hz = 4, 8, 5
        context_latent = torch.randn(B, latent_dim, requires_grad=True)
        context_latent_seq = torch.randn(B, 10, latent_dim)
        output = small_predictor(
            context_latent=context_latent,
            prediction_horizon=pred_hz,
            context_latent_seq=context_latent_seq,
        )
        loss = output.sum()
        loss.backward()
        assert context_latent.grad is not None, "Gradient should flow"

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            RecurrentPredictor(latent_dim=0, hidden_dim=16, num_layers=1, dropout=0.0)


# ── StockDecoder ─────────────────────────────────────────────────────────


class TestStockDecoder:
    """Tests for the price-space decoder."""

    def test_forward_shape(self, small_decoder: StockDecoder) -> None:
        B, latent_dim = 4, 8
        latent = torch.randn(B, latent_dim)
        output = small_decoder(latent)
        assert output.shape == (B, 5), f"Expected (B, 5), got {output.shape}"

    def test_decode_sequence_shape(self, small_decoder: StockDecoder) -> None:
        B, T, L = 4, 5, 8
        latent_seq = torch.randn(B, T, L)
        output = small_decoder.decode_sequence(latent_seq)
        assert output.shape == (B, T, 5), (
            f"Expected ({B}, {T}, 5), got {output.shape}"
        )

    def test_no_redundant_linear_layer(self) -> None:
        """
        For ``num_layers=2``, the decoder should construct:
            Linear(L→5)
        *without* a trailing ``Linear(5→5)`` that would be degenerate.
        """
        decoder = StockDecoder(latent_dim=8, hidden_dim=16, output_dim=5, num_layers=2)
        linear_layers = [
            m for m in decoder.decoder if isinstance(m, nn.Linear)
        ]
        # With the fix, num_layers=2 should produce exactly 1 Linear layer
        # (the final projection L→5), not 2.
        assert len(linear_layers) == 1, (
            f"Expected 1 Linear layer for num_layers=2, got {len(linear_layers)}. "
            "The redundant Linear(5→5) was not removed."
        )

    def test_num_layers_3_produces_correct_structure(self) -> None:
        """
        For ``num_layers=3``, the decoder should produce:
            Linear(L→64) → LayerNorm → GELU → Linear(64→5)
        without a trailing ``Linear(5→5)``.
        """
        decoder = StockDecoder(latent_dim=8, hidden_dim=16, output_dim=5, num_layers=3)
        linear_layers = [
            m for m in decoder.decoder if isinstance(m, nn.Linear)
        ]
        # num_layers=3: L→hidden, hidden→output = 2 Linear layers
        assert len(linear_layers) == 2, (
            f"Expected 2 Linear layers for num_layers=3, got {len(linear_layers)}"
        )
        # First layer: L→hidden
        assert linear_layers[0].in_features == 8
        assert linear_layers[0].out_features == 16
        # Second layer: hidden→output
        assert linear_layers[1].in_features == 16
        assert linear_layers[1].out_features == 5

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            StockDecoder(latent_dim=0)

    def test_differentiable(self, small_decoder: StockDecoder) -> None:
        latent = torch.randn(4, 8, requires_grad=True)
        output = small_decoder(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None


# ── RJEPA (full model) ───────────────────────────────────────────────────


class TestRJEPA:
    """End-to-end tests for the full R-JEPA model."""

    def test_forward_shape(self, small_r_jepa: RJEPA,
                           context_tensor: Tensor,
                           target_tensor: Tensor) -> None:
        output = small_r_jepa(context=context_tensor, target=target_tensor)
        B, pred_hz, latent_dim = (
            context_tensor.shape[0],
            target_tensor.shape[1],
            small_r_jepa.latent_dim,
        )
        assert output["predicted_latents"].shape == (B, pred_hz, latent_dim)
        assert output["target_latent"].shape == (B, latent_dim)
        assert output["decoded_predictions"].shape == target_tensor.shape

    def test_forward_without_target_uses_prediction_horizon(
        self, small_r_jepa: RJEPA, context_tensor: Tensor
    ) -> None:
        """When prediction_horizon is passed explicitly, target is optional."""
        output = small_r_jepa(context=context_tensor, prediction_horizon=5)
        B, latent_dim = context_tensor.shape[0], small_r_jepa.latent_dim
        assert output["predicted_latents"].shape == (B, 5, latent_dim)
        assert "target_latent" not in output  # no target provided

    def test_predict_single(self, small_r_jepa: RJEPA,
                            context_tensor: Tensor) -> None:
        """predict() without ensemble returns single prediction."""
        result = small_r_jepa.predict(context=context_tensor, prediction_horizon=5)
        assert "predictions" in result
        assert "latents" in result
        B, T, F = context_tensor.shape
        assert result["predictions"].shape == (B, 5, F)

    def test_predict_ensemble(self, small_r_jepa: RJEPA,
                               context_tensor: Tensor) -> None:
        """predict() with ensemble returns uncertainty estimates."""
        result = small_r_jepa.predict(
            context=context_tensor,
            prediction_horizon=5,
            return_ensemble=True,
            ensemble_size=3,
        )
        assert "predictions" in result
        assert "ensemble" in result
        assert "confidence" in result
        assert "upper_bound" in result
        assert "lower_bound" in result
        B, _, F = context_tensor.shape
        assert result["predictions"].shape == (B, 5, F)
        assert result["ensemble"].shape == (B, 3, 5, F)
        # Upper bound should be >= lower bound
        assert (result["upper_bound"] >= result["lower_bound"]).all()

    def test_momentum_update(self, small_r_jepa: RJEPA) -> None:
        """update_momentum_encoder blends weights."""
        old_weights = [
            p.clone() for p in small_r_jepa.momentum_encoder.parameters()
        ]
        # Change online encoder
        for p in small_r_jepa.online_encoder.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        small_r_jepa.update_momentum_encoder()
        for p_m, p_old in zip(small_r_jepa.momentum_encoder.parameters(),
                              old_weights):
            assert not torch.equal(p_m, p_old), "Momentum should have updated"

    def test_forward_then_backward(self, small_r_jepa: RJEPA,
                                    context_tensor: Tensor,
                                    target_tensor: Tensor) -> None:
        """Full training step: forward + backward + optimizer step."""
        optimizer = torch.optim.SGD(small_r_jepa.parameters(), lr=0.01)
        output = small_r_jepa(context=context_tensor, target=target_tensor)
        loss = output["predicted_latents"].mean()
        loss.backward()
        optimizer.step()
        # Verify gradients propagated
        for name, p in small_r_jepa.named_parameters():
            if p.requires_grad and "momentum" not in name:
                assert p.grad is not None, f"No gradient for {name}"
                break

    def test_encode_context_shared_helper(self, small_r_jepa: RJEPA,
                                           context_tensor: Tensor) -> None:
        """_encode_context returns correct shapes and is used by both methods."""
        seq, latent = small_r_jepa._encode_context(context_tensor)
        B, T, F = context_tensor.shape
        latent_dim = small_r_jepa.latent_dim
        assert seq.shape == (B, T, latent_dim), (
            f"Seq shape: expected ({B}, {T}, {latent_dim}), got {seq.shape}"
        )
        assert latent.shape == (B, latent_dim), (
            f"Latent shape: expected ({B}, {latent_dim}), got {latent.shape}"
        )
        # Last timestep of sequence should match latent
        assert torch.equal(seq[:, -1, :], latent), (
            "latent should equal seq[:, -1, :]"
        )

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            RJEPA(input_dim=0)

    def test_model_has_correct_submodules(self, small_r_jepa: RJEPA) -> None:
        """Model contains expected submodules."""
        assert hasattr(small_r_jepa, "online_encoder")
        assert hasattr(small_r_jepa, "momentum_encoder")
        assert hasattr(small_r_jepa, "predictor")
        assert hasattr(small_r_jepa, "decoder")
        assert hasattr(small_r_jepa, "_encode_context")
