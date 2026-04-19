"""Tests for the three new diagnostic metrics added in the v5 diagnostics update.

Covers:
    - compute_grad_zero_fraction (He 2603.21173 OCP Thm 1)
    - compute_churn (Tang 2506.00592 C-CHAIN)
    - compute_kfac_hessian_erank (He 2509.22335 Thm 6.2)

Each unit test:
    (a) asserts the expected key set is returned
    (b) asserts no NaN/Inf values
    (c) asserts values are in physically meaningful ranges

Integration tests:
    - DIAGNOSE_KFAC=True: new KFAC keys appear alongside existing diagnostic keys.
    - DIAGNOSE_KFAC=False (default): KFAC keys absent; existing key set unchanged.
    - DIAGNOSE_GRAD_ZERO=True: grad_zero keys present.
    - DIAGNOSE_CHURN=True with snapshots: churn_output key present.
    - DIAGNOSE_CHURN=True without snapshots: churn keys silently absent.
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from gsp_rl.src.actors.diagnostics import (
    compute_grad_zero_fraction,
    compute_churn,
    compute_kfac_hessian_erank,
)
from gsp_rl.src.actors import Actor


# -------------------------------------------------------------------------------------
# Shared test helpers
# -------------------------------------------------------------------------------------

INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 4
BATCH_SIZE = 256


class SmallNet(nn.Module):
    """Minimal Linear+ReLU network for unit testing diagnostic functions."""

    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = HIDDEN_DIM, out_dim: int = OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_batch(n: int = BATCH_SIZE, dim: int = INPUT_DIM) -> torch.Tensor:
    return torch.randn(n, dim)


def _assert_no_nan_inf(d: dict, skip_keys=None):
    skip_keys = skip_keys or set()
    for k, v in d.items():
        if k in skip_keys:
            continue
        assert math.isfinite(v), f"Non-finite value for '{k}': {v}"


# -------------------------------------------------------------------------------------
# Config fixture (reuse the same config.yml as other integration tests)
# -------------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.realpath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "config.yml")

with open(_CONFIG_PATH, "r") as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


def _diag_config(**overrides) -> dict:
    cfg = dict(_BASE_CONFIG)
    cfg["DIAGNOSTICS_ENABLED"] = True
    cfg["DIAGNOSTICS_BATCH_SIZE"] = 64
    cfg["DIAGNOSTICS_FREEZE_EPISODE"] = 0
    cfg["DIAGNOSE_CRITIC"] = False
    cfg.update(overrides)
    return cfg


def _make_actor(network: str, **cfg_overrides) -> Actor:
    config = _diag_config(**cfg_overrides)
    return Actor(
        id=1,
        config=config,
        network=network,
        input_size=8,
        output_size=4,
        min_max_action=1.0,
        meta_param_size=16,
        gsp=False,
        recurrent_gsp=False,
        attention=False,
        gsp_input_size=6,
        gsp_output_size=1,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def _inject_eval_batch(actor: Actor, n: int = 64) -> None:
    input_size = actor.network_input_size
    actor.diag_actor_eval_batch = np.random.randn(n, input_size).astype(np.float32)
    actor.diag_gsp_eval_batch = None


# =====================================================================================
# Unit tests: compute_grad_zero_fraction
# =====================================================================================


class TestComputeGradZeroFraction:
    """Unit tests for compute_grad_zero_fraction."""

    def test_returns_expected_keys(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1", "fc2"])
        assert "grad_zero_fc1" in result
        assert "grad_zero_fc2" in result
        assert len(result) == 2

    def test_no_nan_inf(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1", "fc2"])
        _assert_no_nan_inf(result)

    def test_values_in_unit_interval(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1", "fc2"])
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"grad_zero value out of [0,1]: {k}={v}"

    def test_all_zero_output_produces_large_grad_zero(self):
        """A network with all-zero weights has trivial gradients — grad_zero should be high."""
        net = SmallNet()
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1"])
        # With zero weights the output is always zero, target is zero, loss=0 →
        # gradients are zero everywhere. Fraction of near-zero grads = 1.0.
        assert result["grad_zero_fc1"] == pytest.approx(1.0, abs=0.05)

    def test_net_restored_to_eval_after(self):
        """compute_grad_zero_fraction must leave the network in eval mode."""
        net = SmallNet()
        net.eval()
        batch = _make_batch()
        compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1"])
        assert not net.training

    def test_grad_zero_after_call_is_clear(self):
        """Gradients must be zeroed after the call — no lingering .grad tensors."""
        net = SmallNet()
        batch = _make_batch()
        compute_grad_zero_fraction(net, F.mse_loss, batch, ["fc1", "fc2"])
        for name, p in net.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() == 0.0, (
                    f"Non-zero grad remaining on {name} after compute_grad_zero_fraction"
                )

    def test_missing_layer_name_returns_nan(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["nonexistent"])
        assert math.isnan(result["grad_zero_nonexistent"])

    def test_dotted_path_resolved(self):
        """Dotted layer names (e.g. 'actor.fc1') are resolved correctly."""

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = SmallNet()

            def forward(self, x):
                return self.actor(x)

        net = Wrapper()
        batch = _make_batch()
        result = compute_grad_zero_fraction(net, F.mse_loss, batch, ["actor.fc1"])
        assert "grad_zero_actor_fc1" in result
        assert math.isfinite(result["grad_zero_actor_fc1"])


# =====================================================================================
# Unit tests: compute_churn
# =====================================================================================


class TestComputeChurn:
    """Unit tests for compute_churn."""

    def test_returns_churn_output_key(self):
        net = SmallNet()
        before = copy.deepcopy(net.state_dict())
        # Simulate a training step by perturbing weights
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        after = copy.deepcopy(net.state_dict())

        batch = _make_batch()
        result = compute_churn(net, batch, before, after)
        assert "churn_output" in result

    def test_no_nan_inf(self):
        net = SmallNet()
        before = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        after = copy.deepcopy(net.state_dict())

        batch = _make_batch()
        result = compute_churn(net, batch, before, after)
        _assert_no_nan_inf(result)

    def test_churn_nonnegative(self):
        net = SmallNet()
        before = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        after = copy.deepcopy(net.state_dict())

        batch = _make_batch()
        result = compute_churn(net, batch, before, after)
        for k, v in result.items():
            assert v >= 0.0, f"Churn value negative: {k}={v}"

    def test_zero_churn_when_weights_unchanged(self):
        """Identical before/after state_dicts must produce churn_output ≈ 0."""
        net = SmallNet()
        sd = copy.deepcopy(net.state_dict())
        batch = _make_batch()
        result = compute_churn(net, batch, sd, sd)
        assert result["churn_output"] == pytest.approx(0.0, abs=1e-6)

    def test_per_layer_churn_keys_when_layer_names_given(self):
        net = SmallNet()
        before = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        after = copy.deepcopy(net.state_dict())

        batch = _make_batch()
        result = compute_churn(net, batch, before, after, layer_names=["fc1", "fc2"])
        assert "churn_output" in result
        assert "churn_fc1" in result
        assert "churn_fc2" in result
        _assert_no_nan_inf(result)

    def test_larger_update_produces_larger_churn(self):
        """A bigger weight perturbation should produce higher churn."""
        net = SmallNet()
        batch = _make_batch()

        before = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.001)
        after_small = copy.deepcopy(net.state_dict())

        before2 = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 10.0)
        after_large = copy.deepcopy(net.state_dict())

        churn_small = compute_churn(net, batch, before, after_small)["churn_output"]
        churn_large = compute_churn(net, batch, before2, after_large)["churn_output"]
        assert churn_large > churn_small


# =====================================================================================
# Unit tests: compute_kfac_hessian_erank
# =====================================================================================


class TestComputeKfacHessianErank:
    """Unit tests for compute_kfac_hessian_erank."""

    def test_returns_expected_keys(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_kfac_hessian_erank(net, batch, ["fc1", "fc2"])
        assert "kfac_erank_fc1" in result
        assert "kfac_erank_fc2" in result
        assert "kfac_erank_A_fc1" in result
        assert "kfac_erank_G_fc1" in result
        assert "kfac_erank_A_fc2" in result
        assert "kfac_erank_G_fc2" in result

    def test_no_nan_inf(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_kfac_hessian_erank(net, batch, ["fc1", "fc2"])
        _assert_no_nan_inf(result)

    def test_kfac_erank_at_least_one(self):
        """Effective rank must be >= 1 for a non-degenerate network."""
        net = SmallNet()
        batch = _make_batch()
        result = compute_kfac_hessian_erank(net, batch, ["fc1", "fc2"])
        for key in ("kfac_erank_fc1", "kfac_erank_fc2"):
            assert result[key] >= 1.0, f"{key}={result[key]} is below 1"

    def test_product_equals_a_times_g(self):
        """kfac_erank should equal kfac_erank_A * kfac_erank_G."""
        net = SmallNet()
        batch = _make_batch()
        result = compute_kfac_hessian_erank(net, batch, ["fc1"])
        expected = result["kfac_erank_A_fc1"] * result["kfac_erank_G_fc1"]
        assert result["kfac_erank_fc1"] == pytest.approx(expected, abs=1e-6)

    def test_net_restored_to_eval_after(self):
        """compute_kfac_hessian_erank must leave the network in eval mode."""
        net = SmallNet()
        net.eval()
        batch = _make_batch()
        compute_kfac_hessian_erank(net, batch, ["fc1"])
        assert not net.training

    def test_grad_zero_after_call(self):
        """Gradients must be zeroed after the call."""
        net = SmallNet()
        batch = _make_batch()
        compute_kfac_hessian_erank(net, batch, ["fc1"])
        for name, p in net.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() == 0.0, (
                    f"Non-zero grad remaining on {name} after compute_kfac_hessian_erank"
                )

    def test_missing_layer_returns_nan(self):
        net = SmallNet()
        batch = _make_batch()
        result = compute_kfac_hessian_erank(net, batch, ["nonexistent"])
        assert math.isnan(result["kfac_erank_nonexistent"])

    def test_erank_bounded_by_layer_dims(self):
        """The A covariance rank is bounded by min(N, d_in) and G by min(N, d_out)."""
        net = SmallNet(in_dim=INPUT_DIM, hidden=HIDDEN_DIM, out_dim=OUTPUT_DIM)
        batch = _make_batch(n=BATCH_SIZE, dim=INPUT_DIM)
        result = compute_kfac_hessian_erank(net, batch, ["fc1"])
        # A_fc1 is (INPUT_DIM x INPUT_DIM), rank <= min(BATCH_SIZE, INPUT_DIM)
        assert result["kfac_erank_A_fc1"] <= min(BATCH_SIZE, INPUT_DIM) + 1  # +1 for fp tolerance
        # G_fc1 is (HIDDEN_DIM x HIDDEN_DIM), rank <= min(BATCH_SIZE, HIDDEN_DIM)
        assert result["kfac_erank_G_fc1"] <= min(BATCH_SIZE, HIDDEN_DIM) + 1


# =====================================================================================
# Integration tests: Actor.compute_diagnostics with new flags
# =====================================================================================


class TestActorComputeDiagnosticsNewMetrics:
    """Integration tests: new metrics appear in compute_diagnostics when flags are set."""

    def test_diagnose_kfac_true_adds_kfac_keys(self):
        """When DIAGNOSE_KFAC=True, kfac_erank_* keys appear in compute_diagnostics."""
        actor = _make_actor("DDPG", DIAGNOSE_KFAC=True, DIAGNOSE_GRAD_ZERO=False)
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()

        # Existing keys must still be present
        assert "diag_actor_fau_fc1" in result
        assert "diag_actor_fau_fc2" in result
        assert "diag_actor_erank_penult" in result

        # New KFAC keys must be present
        kfac_keys = [k for k in result if "kfac_erank" in k]
        assert len(kfac_keys) > 0, "No kfac_erank keys found in result"
        for k in kfac_keys:
            assert math.isfinite(result[k]), f"Non-finite KFAC key: {k}={result[k]}"

    def test_diagnose_kfac_false_default_no_kfac_keys(self):
        """When DIAGNOSE_KFAC=False (default), no kfac_erank_* keys appear."""
        actor = _make_actor("DDPG")  # DIAGNOSE_KFAC defaults to False
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()

        kfac_keys = [k for k in result if "kfac_erank" in k]
        assert kfac_keys == [], f"Unexpected KFAC keys when flag is off: {kfac_keys}"

    def test_diagnose_grad_zero_true_adds_grad_zero_keys(self):
        """When DIAGNOSE_GRAD_ZERO=True, grad_zero_* keys appear."""
        actor = _make_actor("DDPG", DIAGNOSE_GRAD_ZERO=True)
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()

        grad_keys = [k for k in result if "grad_zero" in k]
        assert len(grad_keys) > 0, "No grad_zero keys found when DIAGNOSE_GRAD_ZERO=True"
        for k in grad_keys:
            assert math.isfinite(result[k]), f"Non-finite grad_zero key: {k}={result[k]}"
            assert 0.0 <= result[k] <= 1.0, f"grad_zero out of [0,1]: {k}={result[k]}"

    def test_diagnose_grad_zero_false_no_grad_zero_keys(self):
        """When DIAGNOSE_GRAD_ZERO=False, no grad_zero_* keys appear."""
        actor = _make_actor("DDPG", DIAGNOSE_GRAD_ZERO=False)
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()

        grad_keys = [k for k in result if "grad_zero" in k]
        assert grad_keys == [], f"Unexpected grad_zero keys: {grad_keys}"

    def test_diagnose_churn_with_snapshots_adds_churn_key(self):
        """When DIAGNOSE_CHURN=True and snapshots provided, churn_output appears."""
        actor = _make_actor("DDPG", DIAGNOSE_CHURN=True, DIAGNOSE_GRAD_ZERO=False)
        _inject_eval_batch(actor)

        main_net = actor._main_network(actor.networks)
        before_sd = copy.deepcopy(main_net.state_dict())
        with torch.no_grad():
            for p in main_net.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        after_sd = copy.deepcopy(main_net.state_dict())

        result = actor.compute_diagnostics(
            actor_before_state_dict=before_sd,
            actor_after_state_dict=after_sd,
        )

        churn_keys = [k for k in result if "churn" in k]
        assert len(churn_keys) > 0, "No churn keys found when snapshots provided"
        for k in churn_keys:
            assert math.isfinite(result[k]), f"Non-finite churn key: {k}={result[k]}"
            assert result[k] >= 0.0, f"Negative churn: {k}={result[k]}"

    def test_diagnose_churn_without_snapshots_no_churn_key(self):
        """When DIAGNOSE_CHURN=True but no snapshots provided, churn is silently absent."""
        actor = _make_actor("DDPG", DIAGNOSE_CHURN=True, DIAGNOSE_GRAD_ZERO=False)
        _inject_eval_batch(actor)

        result = actor.compute_diagnostics()  # No snapshots passed
        churn_keys = [k for k in result if "churn" in k]
        assert churn_keys == [], (
            f"Churn keys present without snapshots: {churn_keys}"
        )

    def test_backward_compat_ddqn_key_set_unchanged_when_new_flags_off(self):
        """With all new flags OFF, DDQN key set matches historical schema exactly."""
        actor = _make_actor(
            "DDQN",
            DIAGNOSE_GRAD_ZERO=False,
            DIAGNOSE_CHURN=False,
            DIAGNOSE_KFAC=False,
        )
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()

        expected_historical_keys = {
            "diag_actor_fau_fc1",
            "diag_actor_fau_fc2",
            "diag_actor_overactive_fc1",
            "diag_actor_overactive_fc2",
            "diag_actor_wnorm_fc1",
            "diag_actor_wnorm_fc2",
            "diag_actor_wnorm_fc3",
            "diag_actor_erank_penult",
            "diag_q_action_gap_mean",
            "diag_q_action_gap_std",
            "diag_q_max_mean",
        }
        for key in expected_historical_keys:
            assert key in result, f"Missing historical key: {key}"

        # No new keys should appear
        new_key_prefixes = ("grad_zero", "churn", "kfac_erank")
        unexpected = [k for k in result if any(p in k for p in new_key_prefixes)]
        assert unexpected == [], f"Unexpected new keys when all flags are off: {unexpected}"

    def test_all_new_metrics_enabled_together(self):
        """All three new flags on simultaneously: all key families present, no NaN."""
        actor = _make_actor(
            "DDPG",
            DIAGNOSE_GRAD_ZERO=True,
            DIAGNOSE_CHURN=True,
            DIAGNOSE_KFAC=True,
        )
        _inject_eval_batch(actor)

        main_net = actor._main_network(actor.networks)
        before_sd = copy.deepcopy(main_net.state_dict())
        with torch.no_grad():
            for p in main_net.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        after_sd = copy.deepcopy(main_net.state_dict())

        result = actor.compute_diagnostics(
            actor_before_state_dict=before_sd,
            actor_after_state_dict=after_sd,
        )

        # All three families must have at least one key
        assert any("grad_zero" in k for k in result), "No grad_zero keys"
        assert any("churn" in k for k in result), "No churn keys"
        assert any("kfac_erank" in k for k in result), "No kfac_erank keys"

        # Existing keys still present
        assert "diag_actor_fau_fc1" in result
        assert "diag_actor_erank_penult" in result

        # No NaN/Inf
        for k, v in result.items():
            assert math.isfinite(v), f"Non-finite value: {k}={v}"
