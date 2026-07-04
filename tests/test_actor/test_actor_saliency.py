"""Tests for the actor-usage diagnostics: dQ/d(pred) saliency (M1) and
weight-on-pred relative norm (M3).

These measure whether the actor's Q-network actually *uses* the GSP prediction
dims that are concatenated onto the tail of its augmented input vector. See the
2026-07-04 actor-usage pre-registration.

The GSP prediction occupies the LAST ``self.gsp_network_output`` columns of the
actor's augmented input (actor.py:134-144: ``network_input_size = input_size +
gsp_network_output``). We construct a ``gsp=True`` DDQN actor so that pred slice
exists, then seed the Q-network's first-layer weights to control whether the
output depends on the pred dims or the non-pred dims.

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
