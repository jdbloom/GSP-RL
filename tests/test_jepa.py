"""Tests for the JEPA (Joint Embedding Predictive Architecture) modules.

Covers:
1. test_encoder_predictor_shape   — JEPAEncoder and JEPAPredictor produce the
   expected output shapes for a batch of inputs.
2. test_target_ema_update         — After one EMA step with tau=0.5 and known
   initial weights, the target encoder parameters equal
   0.5 * old_target + 0.5 * online (i.e. arithmetic mean).
"""

import copy

import torch
import pytest

from gsp_rl.src.networks.jepa import JEPAEncoder, JEPAPredictor


INPUT_DIM = 6
LATENT_DIM = 32
BATCH_SIZE = 16
ACTION_DIM = 3


class TestEncoderPredictorShape:
    """Shape contracts for JEPAEncoder and JEPAPredictor."""

    def test_encoder_output_shape(self):
        enc = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        x = torch.randn(BATCH_SIZE, INPUT_DIM).to(enc.device)
        z = enc(x)
        assert z.shape == (BATCH_SIZE, LATENT_DIM), (
            f"Expected encoder output shape ({BATCH_SIZE}, {LATENT_DIM}), got {z.shape}"
        )

    def test_predictor_output_shape(self):
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64)
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        z_pred = pred(z)
        assert z_pred.shape == (BATCH_SIZE, LATENT_DIM), (
            f"Expected predictor output shape ({BATCH_SIZE}, {LATENT_DIM}), got {z_pred.shape}"
        )

    def test_encoder_single_sample(self):
        """Single-sample (batch=1) forward should not crash."""
        enc = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        x = torch.randn(1, INPUT_DIM).to(enc.device)
        z = enc(x)
        assert z.shape == (1, LATENT_DIM)

    def test_encoder_no_nan(self):
        """Encoder output must be finite on random input."""
        enc = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        x = torch.randn(BATCH_SIZE, INPUT_DIM).to(enc.device)
        z = enc(x)
        assert torch.isfinite(z).all(), "Encoder output contains NaN or Inf"

    def test_predictor_no_nan(self):
        """Predictor output must be finite on random latent."""
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64)
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        z_pred = pred(z)
        assert torch.isfinite(z_pred).all(), "Predictor output contains NaN or Inf"


class TestActionConditionedPredictor:
    """Flag GSP_JEPA_ACTION_COND — the predictor consumes (z_t, a_t)."""

    def test_default_predictor_is_not_action_conditioned(self):
        """Back-compat: with action_dim=0 (default) the predictor takes z only."""
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64)
        assert pred.action_dim == 0
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        # Single-arg call still works and returns latent_dim.
        out = pred(z)
        assert out.shape == (BATCH_SIZE, LATENT_DIM)

    def test_action_conditioned_predictor_consumes_action(self):
        """With action_dim>0 the predictor's fc1 input width == latent+action."""
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64, action_dim=ACTION_DIM)
        assert pred.action_dim == ACTION_DIM
        assert pred.fc1.weight.shape[1] == LATENT_DIM + ACTION_DIM
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        a = torch.randn(BATCH_SIZE, ACTION_DIM).to(pred.device)
        out = pred(z, a)
        assert out.shape == (BATCH_SIZE, LATENT_DIM)

    def test_action_conditioned_predictor_requires_action(self):
        """Action-conditioned predictor must receive an action tensor."""
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64, action_dim=ACTION_DIM)
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        with pytest.raises((TypeError, ValueError)):
            pred(z)  # missing required action

    def test_action_changes_prediction(self):
        """Different actions must produce different predicted futures."""
        pred = JEPAPredictor(latent_dim=LATENT_DIM, hidden=64, action_dim=ACTION_DIM)
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(pred.device)
        a1 = torch.zeros(BATCH_SIZE, ACTION_DIM).to(pred.device)
        a2 = torch.ones(BATCH_SIZE, ACTION_DIM).to(pred.device)
        out1 = pred(z, a1)
        out2 = pred(z, a2)
        assert not torch.allclose(out1, out2), (
            "Action-conditioned predictor ignored the action input"
        )


class TestCosineLatentLoss:
    """Flag GSP_JEPA_COSINE_LOSS — normalized/cosine latent loss (SPR 2007.05929)."""

    def test_cosine_loss_zero_on_identical(self):
        """Cosine loss between a latent and itself is ~0 (perfect similarity)."""
        from gsp_rl.src.actors.learning_aids import jepa_cosine_loss
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss = jepa_cosine_loss(z, z.clone())
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), (
            f"Cosine loss on identical latents should be 0, got {loss.item()}"
        )

    def test_cosine_loss_scale_invariant(self):
        """Cosine loss is invariant to positive scaling of either argument."""
        from gsp_rl.src.actors.learning_aids import jepa_cosine_loss
        pred = torch.randn(BATCH_SIZE, LATENT_DIM)
        target = torch.randn(BATCH_SIZE, LATENT_DIM)
        base = jepa_cosine_loss(pred, target)
        scaled = jepa_cosine_loss(pred * 7.0, target * 0.3)
        assert torch.isclose(base, scaled, atol=1e-5), (
            "Cosine loss must be invariant to positive rescaling of inputs"
        )

    def test_cosine_loss_max_on_opposite(self):
        """Cosine loss between opposite vectors is ~2 (1 - (-1))."""
        from gsp_rl.src.actors.learning_aids import jepa_cosine_loss
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss = jepa_cosine_loss(z, -z)
        assert torch.isclose(loss, torch.tensor(2.0), atol=1e-5), (
            f"Cosine loss on opposite latents should be 2, got {loss.item()}"
        )

    def test_cosine_loss_differentiable(self):
        """Cosine loss must produce a gradient on the prediction argument."""
        from gsp_rl.src.actors.learning_aids import jepa_cosine_loss
        pred = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        target = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss = jepa_cosine_loss(pred, target)
        loss.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


class TestTargetEmaUpdate:
    """EMA update correctness for the target encoder."""

    def test_ema_tau_half(self):
        """With tau=0.5, after one EMA step:
           target_p = 0.5 * old_target + 0.5 * online_p
        which equals the arithmetic mean of old and online weights.
        """
        tau = 0.5
        enc_online = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        enc_target = copy.deepcopy(enc_online)

        # Perturb online weights so online != target
        with torch.no_grad():
            for p in enc_online.parameters():
                p.add_(torch.ones_like(p) * 2.0)  # shift online by +2

        # Capture old target weights and online weights
        old_target_params = {name: p.data.clone() for name, p in enc_target.named_parameters()}
        online_params = {name: p.data.clone() for name, p in enc_online.named_parameters()}

        # Perform EMA update
        with torch.no_grad():
            for online_p, target_p in zip(
                enc_online.parameters(), enc_target.parameters()
            ):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

        # Verify: each target param = 0.5 * old_target + 0.5 * online
        for name, target_p in enc_target.named_parameters():
            expected = 0.5 * old_target_params[name] + 0.5 * online_params[name]
            assert torch.allclose(target_p.data, expected, atol=1e-6), (
                f"EMA mismatch at param '{name}': "
                f"expected mean of old+online, got divergent values"
            )

    def test_ema_tau_one_freezes_target(self):
        """With tau=1.0, the target should not change (fully frozen update)."""
        tau = 1.0
        enc_online = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        enc_target = copy.deepcopy(enc_online)

        # Snapshot target before update
        old_target_params = {name: p.data.clone() for name, p in enc_target.named_parameters()}

        # Perturb online
        with torch.no_grad():
            for p in enc_online.parameters():
                p.add_(torch.ones_like(p) * 5.0)

        # EMA with tau=1.0
        with torch.no_grad():
            for online_p, target_p in zip(
                enc_online.parameters(), enc_target.parameters()
            ):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

        for name, target_p in enc_target.named_parameters():
            assert torch.allclose(target_p.data, old_target_params[name], atol=1e-6), (
                f"With tau=1.0, target param '{name}' should be unchanged"
            )

    def test_ema_tau_zero_copies_online(self):
        """With tau=0.0, the target should become a copy of online."""
        tau = 0.0
        enc_online = JEPAEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden=128)
        enc_target = copy.deepcopy(enc_online)

        # Perturb online
        with torch.no_grad():
            for p in enc_online.parameters():
                p.add_(torch.ones_like(p) * 3.0)

        online_params = {name: p.data.clone() for name, p in enc_online.named_parameters()}

        # EMA with tau=0.0
        with torch.no_grad():
            for online_p, target_p in zip(
                enc_online.parameters(), enc_target.parameters()
            ):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

        for name, target_p in enc_target.named_parameters():
            assert torch.allclose(target_p.data, online_params[name], atol=1e-6), (
                f"With tau=0.0, target param '{name}' should equal online"
            )
