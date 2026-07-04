"""Tests for the actor-usage diagnostics: dQ/d(pred) saliency (M1) and
weight-on-pred relative norm (M3).

These measure whether the actor's Q-network actually *uses* the GSP prediction
dims concatenated into its augmented input vector. See the 2026-07-04
actor-usage pre-registration.

The GSP prediction begins at OFFSET ``self.input_size`` (the raw env_obs width,
actor.py:79) and has width ``self.gsp_network_output`` (or ``gsp_encoder_dim``
under JEPA). It is an offset slice, NOT a tail slice: make_agent_state
concatenates ``(env_obs, gsp_slot, global_knowledge)``, so trailing columns
(global_knowledge) may follow the prediction and the pred sits in the MIDDLE.
We construct a ``gsp=True`` DDQN actor so the pred slice exists, then seed the
Q-network's first-layer weights to control whether the output depends on the
pred dims, the non-pred obs dims, or the trailing dims.

M1 (``diag_gsp_actor_saliency``): mean |dQ/dx| over pred dims divided by mean
    |dQ/dx| over non-pred dims. >> 1 means the actor's Q-value is driven by the
    prediction; ~0 means the prediction is ignored.
M3 (``diag_gsp_actor_wnorm_pred_rel``): mean first-layer column norm over pred
    dims relative to the mean column norm across all input dims.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
import yaml

from gsp_rl.src.actors import Actor


_HERE = os.path.dirname(os.path.realpath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "config.yml")

with open(_CONFIG_PATH, "r") as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


# Base observation dims and pred (GSP output) dims. Augmented input =
# INPUT_SIZE + PRED_WIDTH; pred slice = last PRED_WIDTH columns.
INPUT_SIZE = 8
PRED_WIDTH = 3
OUTPUT_SIZE = 4


def _diag_config(**overrides) -> dict:
    cfg = dict(_BASE_CONFIG)
    cfg["DIAGNOSTICS_ENABLED"] = True
    cfg["DIAGNOSTICS_BATCH_SIZE"] = 64
    cfg["DIAGNOSTICS_FREEZE_EPISODE"] = 0
    cfg["DIAGNOSE_CRITIC"] = False
    cfg.update(overrides)
    return cfg


def _make_gsp_ddqn_actor(**cfg_overrides) -> Actor:
    """DDQN actor with GSP enabled so the actor's augmented input carries a
    pred slice of width ``PRED_WIDTH`` in its last columns."""
    config = _diag_config(**cfg_overrides)
    return Actor(
        id=1,
        config=config,
        network="DDQN",
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1.0,
        meta_param_size=16,
        gsp=True,
        recurrent_gsp=False,
        attention=False,
        gsp_input_size=6,
        gsp_output_size=PRED_WIDTH,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def _inject_eval_batch(actor: Actor, n: int = 64) -> None:
    input_size = actor.network_input_size
    actor.diag_actor_eval_batch = np.random.randn(n, input_size).astype(np.float32)
    actor.diag_gsp_eval_batch = None


def _q_net(actor: Actor):
    return actor._main_network(actor.networks)


# =====================================================================================
# Setup sanity: the actor's augmented input carries a pred slice of PRED_WIDTH.
# =====================================================================================


def test_pred_slice_width_matches_gsp_output():
    """Augmented input width == INPUT_SIZE + PRED_WIDTH; pred is the tail slice."""
    actor = _make_gsp_ddqn_actor()
    assert actor.network_input_size == INPUT_SIZE + PRED_WIDTH
    assert actor.gsp_network_output == PRED_WIDTH


# =====================================================================================
# M1: diag_gsp_actor_saliency
# =====================================================================================


class TestActorSaliency:
    def test_saliency_high_when_output_depends_only_on_pred(self):
        """Zero the non-pred input columns of fc1 → Q depends ONLY on pred dims →
        saliency ratio >> 1."""
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        net = _q_net(actor)
        with torch.no_grad():
            W = net.fc1.weight  # [hidden, in]
            W.zero_()
            # Give the pred columns (last PRED_WIDTH) real weight.
            W[:, INPUT_SIZE:] = torch.randn_like(W[:, INPUT_SIZE:])

        result = actor.compute_diagnostics()
        assert "diag_gsp_actor_saliency" in result
        assert result["diag_gsp_actor_saliency"] > 1.0
        assert result["diag_gsp_actor_saliency_abs"] > 0.0

    def test_saliency_near_zero_when_pred_weights_zeroed(self):
        """Zero the pred-column weights of fc1 → Q ignores pred dims →
        saliency ratio ~0."""
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        net = _q_net(actor)
        with torch.no_grad():
            W = net.fc1.weight
            # Non-pred columns carry all the signal; pred columns are dead.
            W[:, :INPUT_SIZE] = torch.randn_like(W[:, :INPUT_SIZE])
            W[:, INPUT_SIZE:].zero_()

        result = actor.compute_diagnostics()
        assert "diag_gsp_actor_saliency" in result
        assert result["diag_gsp_actor_saliency"] < 0.1

    def test_saliency_keys_finite(self):
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()
        assert math.isfinite(result["diag_gsp_actor_saliency"])
        assert math.isfinite(result["diag_gsp_actor_saliency_abs"])

    def test_saliency_absent_when_gsp_disabled(self):
        """No pred slice exists without GSP → saliency keys absent."""
        config = _diag_config()
        actor = Actor(
            id=1, config=config, network="DDQN", input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE, min_max_action=1.0, meta_param_size=16,
            gsp=False, recurrent_gsp=False, attention=False, gsp_input_size=6,
            gsp_output_size=1, gsp_look_back=2, gsp_sequence_length=5,
        )
        input_size = actor.network_input_size
        actor.diag_actor_eval_batch = np.random.randn(64, input_size).astype(np.float32)
        actor.diag_gsp_eval_batch = None
        result = actor.compute_diagnostics()
        assert "diag_gsp_actor_saliency" not in result
        assert "diag_gsp_actor_wnorm_pred_rel" not in result

    def test_saliency_does_not_leak_requires_grad(self):
        """The stored eval batch must not retain requires_grad after diagnostics."""
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        _ = actor.compute_diagnostics()
        # The stored eval batch is a numpy array; it must remain numpy and clean.
        assert isinstance(actor.diag_actor_eval_batch, np.ndarray)


# =====================================================================================
# M3: diag_gsp_actor_wnorm_pred_rel
# =====================================================================================


class TestActorWnormPredRel:
    def test_wnorm_pred_rel_scales_with_pred_column_norm(self):
        """Set pred columns to have 10x the norm of the non-pred columns →
        wnorm_pred_rel ≈ 10 / (weighted mean)."""
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        net = _q_net(actor)
        in_dim = actor.network_input_size
        hidden = net.fc1.weight.shape[0]
        with torch.no_grad():
            W = net.fc1.weight
            # Give every non-pred column unit L2 norm, every pred column 10x.
            base = torch.ones(hidden) / math.sqrt(hidden)  # column of unit norm
            for c in range(in_dim):
                scale = 10.0 if c >= INPUT_SIZE else 1.0
                W[:, c] = base * scale

        result = actor.compute_diagnostics()
        assert "diag_gsp_actor_wnorm_pred_rel" in result
        # col_norm: non-pred = 1.0 (INPUT_SIZE of them), pred = 10.0 (PRED_WIDTH).
        mean_norm = (INPUT_SIZE * 1.0 + PRED_WIDTH * 10.0) / in_dim
        expected = 10.0 / (mean_norm + 1e-8)
        assert result["diag_gsp_actor_wnorm_pred_rel"] == pytest.approx(expected, rel=1e-3)

    def test_wnorm_pred_rel_equals_one_when_uniform(self):
        """Uniform column norms → ratio == 1."""
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        net = _q_net(actor)
        in_dim = actor.network_input_size
        hidden = net.fc1.weight.shape[0]
        with torch.no_grad():
            base = torch.ones(hidden) / math.sqrt(hidden)
            for c in range(in_dim):
                net.fc1.weight[:, c] = base
        result = actor.compute_diagnostics()
        assert result["diag_gsp_actor_wnorm_pred_rel"] == pytest.approx(1.0, rel=1e-4)

    def test_wnorm_pred_rel_finite(self):
        actor = _make_gsp_ddqn_actor()
        _inject_eval_batch(actor)
        result = actor.compute_diagnostics()
        assert math.isfinite(result["diag_gsp_actor_wnorm_pred_rel"])


# =====================================================================================
# Regression: OFFSET slice (not tail) when trailing dims follow the prediction.
#
# This is the global_knowledge case: make_agent_state concatenates
# (env_obs, gsp_slot, global_knowledge), so the augmented input layout is
# [obs (input_size) | pred (gsp_network_output) | trailing (global_knowledge)].
# A tail-based slice would measure the trailing block instead of the pred block.
# =====================================================================================

TRAILING = 5  # simulated global_knowledge width


def _widen_qnet_fc1_with_trailing(actor: Actor) -> None:
    """Rebuild the Q-network's fc1 so its input width is
    input_size + PRED_WIDTH + TRAILING, simulating [obs | pred | trailing].

    The actor's book-kept self.input_size (offset) and gsp_network_output (pred
    width) are unchanged, so compute_diagnostics derives
    pred_slice = slice(INPUT_SIZE, INPUT_SIZE + PRED_WIDTH) — the MIDDLE block.
    """
    net = _q_net(actor)
    hidden = net.fc1.weight.shape[0]
    device = net.fc1.weight.device
    new_in = INPUT_SIZE + PRED_WIDTH + TRAILING
    new_fc1 = torch.nn.Linear(new_in, hidden).to(device)
    with torch.no_grad():
        new_fc1.weight.zero_()
        new_fc1.bias.zero_()
    net.fc1 = new_fc1


def _inject_wide_eval_batch(actor: Actor, n: int = 64) -> None:
    width = INPUT_SIZE + PRED_WIDTH + TRAILING
    actor.diag_actor_eval_batch = np.random.randn(n, width).astype(np.float32)
    actor.diag_gsp_eval_batch = None


# Column index helpers for the [obs | pred | trailing] layout.
_PRED_LO, _PRED_HI = INPUT_SIZE, INPUT_SIZE + PRED_WIDTH
_TRAIL_LO, _TRAIL_HI = INPUT_SIZE + PRED_WIDTH, INPUT_SIZE + PRED_WIDTH + TRAILING


class TestOffsetSliceWithTrailingDims:
    def test_high_saliency_when_q_depends_only_on_pred_block(self):
        """Q depends ONLY on the middle pred block → high pred saliency, even
        though there are trailing (global_knowledge) columns after it."""
        actor = _make_gsp_ddqn_actor()
        _widen_qnet_fc1_with_trailing(actor)
        _inject_wide_eval_batch(actor)
        net = _q_net(actor)
        with torch.no_grad():
            net.fc1.weight[:, _PRED_LO:_PRED_HI] = torch.randn_like(
                net.fc1.weight[:, _PRED_LO:_PRED_HI]
            )
        result = actor.compute_diagnostics()
        assert "diag_gsp_actor_saliency" in result
        assert result.get("diag_gsp_actor_saliency_nan", 0.0) == 0.0
        assert result["diag_gsp_actor_saliency"] > 1.0

    def test_low_saliency_when_q_depends_only_on_trailing_block(self):
        """Q depends ONLY on the trailing (global_knowledge) block → pred
        saliency must be LOW. A tail slice would WRONGLY report this as high
        because it would measure the trailing columns. This is the regression
        that catches the tail bug."""
        actor = _make_gsp_ddqn_actor()
        _widen_qnet_fc1_with_trailing(actor)
        _inject_wide_eval_batch(actor)
        net = _q_net(actor)
        with torch.no_grad():
            net.fc1.weight[:, _TRAIL_LO:_TRAIL_HI] = torch.randn_like(
                net.fc1.weight[:, _TRAIL_LO:_TRAIL_HI]
            )
        result = actor.compute_diagnostics()
        assert result.get("diag_gsp_actor_saliency_nan", 0.0) == 0.0
        assert result["diag_gsp_actor_saliency"] < 0.1

    def test_wnorm_offset_picks_pred_not_trailing(self):
        """M3 column-norm ratio must key off the middle pred block, not trailing."""
        actor = _make_gsp_ddqn_actor()
        _widen_qnet_fc1_with_trailing(actor)
        _inject_wide_eval_batch(actor)
        net = _q_net(actor)
        hidden = net.fc1.weight.shape[0]
        in_dim = INPUT_SIZE + PRED_WIDTH + TRAILING
        with torch.no_grad():
            unit = torch.ones(hidden) / math.sqrt(hidden)  # unit-norm column
            for c in range(in_dim):
                # Pred columns get 10x; obs AND trailing columns get 1x.
                scale = 10.0 if (_PRED_LO <= c < _PRED_HI) else 1.0
                net.fc1.weight[:, c] = unit * scale
        result = actor.compute_diagnostics()
        # col_norm: PRED_WIDTH cols at 10.0, the rest (INPUT_SIZE + TRAILING) at 1.0.
        mean_norm = (PRED_WIDTH * 10.0 + (INPUT_SIZE + TRAILING) * 1.0) / in_dim
        expected = 10.0 / (mean_norm + 1e-8)
        assert result["diag_gsp_actor_wnorm_pred_rel"] == pytest.approx(expected, rel=1e-3)


class TestOffsetGuard:
    def test_guard_emits_nan_sentinel_when_slot_cannot_fit(self):
        """If fc1's input width is too small to hold [obs | pred] at the expected
        offset, emit the NaN sentinel instead of mismeasuring."""
        actor = _make_gsp_ddqn_actor()
        net = _q_net(actor)
        hidden = net.fc1.weight.shape[0]
        # Shrink fc1 so input width < input_size + pred_width (base + pred > in_dim).
        narrow_in = INPUT_SIZE  # no room for the pred block at offset INPUT_SIZE
        device = net.fc1.weight.device
        with torch.no_grad():
            net.fc1 = torch.nn.Linear(narrow_in, hidden).to(device)
        actor.diag_actor_eval_batch = np.random.randn(64, narrow_in).astype(np.float32)
        actor.diag_gsp_eval_batch = None
        result = actor.compute_diagnostics()
        assert result["diag_gsp_actor_saliency"] == 0.0
        assert result["diag_gsp_actor_saliency_nan"] == 1.0
        # Must NOT have computed a (wrong) wnorm ratio for the missing slot.
        assert "diag_gsp_actor_wnorm_pred_rel" not in result
