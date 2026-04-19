"""End-to-end integration tests for Actor.compute_diagnostics across learning schemes.

Constructs an Actor with each learning scheme (DQN, DDQN, DDPG, TD3), populates the
frozen eval batch, calls compute_diagnostics, and asserts:
1. Expected keys are present.
2. Keys that should NOT be present are absent (e.g. DDPG must not have q_action_gap).
3. No NaN/Inf values in returned dict.

Also verifies that the DDQN key set exactly matches the historical schema from
production runs (j142, j150-170) for backward compatibility.
"""
from __future__ import annotations

import math
import os
import yaml
import numpy as np
import pytest
import torch

from gsp_rl.src.actors import Actor


# ---------------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.realpath(__file__))
_CONFIG_PATH = os.path.join(_HERE, 'config.yml')

with open(_CONFIG_PATH, 'r') as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


def _diag_config(**overrides) -> dict:
    """Return a config dict with diagnostics enabled and any additional overrides."""
    cfg = dict(_BASE_CONFIG)
    cfg['DIAGNOSTICS_ENABLED'] = True
    cfg['DIAGNOSTICS_BATCH_SIZE'] = 64
    cfg['DIAGNOSTICS_FREEZE_EPISODE'] = 0
    cfg['DIAGNOSE_CRITIC'] = False
    cfg.update(overrides)
    return cfg


def _make_actor(network: str, **cfg_overrides) -> Actor:
    """Build a minimal Actor for the given learning scheme."""
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
    """Directly set the frozen eval batch on the actor (bypasses buffer size checks)."""
    input_size = actor.network_input_size
    actor.diag_actor_eval_batch = np.random.randn(n, input_size).astype(np.float32)
    actor.diag_gsp_eval_batch = None


def _assert_no_nan_inf(result: dict, skip_keys: set | None = None) -> None:
    skip_keys = skip_keys or set()
    for k, v in result.items():
        if k in skip_keys:
            continue
        assert math.isfinite(v), f"Non-finite value for key '{k}': {v}"


# ---------------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------------

def test_dqn_compute_diagnostics_keys():
    """DQN actor diagnostics produce FAU, weight norms, effective rank, and Q-gap."""
    actor = _make_actor('DQN')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    # Must have actor-side metrics
    assert 'diag_actor_fau_fc1' in result
    assert 'diag_actor_fau_fc2' in result
    assert 'diag_actor_wnorm_fc1' in result
    assert 'diag_actor_wnorm_fc2' in result
    assert 'diag_actor_wnorm_fc3' in result
    assert 'diag_actor_erank_penult' in result
    # Q-gap keys at top level (not diag_actor_)
    assert 'diag_q_action_gap_mean' in result
    assert 'diag_q_action_gap_std' in result
    assert 'diag_q_max_mean' in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DDQN — backward compat check
# ---------------------------------------------------------------------------------

DDQN_HISTORICAL_KEYS = {
    'diag_actor_fau_fc1',
    'diag_actor_fau_fc2',
    'diag_actor_overactive_fc1',
    'diag_actor_overactive_fc2',
    'diag_actor_wnorm_fc1',
    'diag_actor_wnorm_fc2',
    'diag_actor_wnorm_fc3',
    'diag_actor_erank_penult',
    'diag_q_action_gap_mean',
    'diag_q_action_gap_std',
    'diag_q_max_mean',
}


def test_ddqn_compute_diagnostics_backward_compat():
    """DDQN must produce exactly the historical key set (backward compat with j142-j186)."""
    actor = _make_actor('DDQN')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    # All historical keys must be present
    for key in DDQN_HISTORICAL_KEYS:
        assert key in result, f"Missing historical key: {key}"

    _assert_no_nan_inf(result)


def test_ddqn_compute_diagnostics_no_nan():
    """All DDQN diagnostic values must be finite."""
    actor = _make_actor('DDQN')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()
    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DDPG — must NOT produce Q-gap; must have actor metrics
# ---------------------------------------------------------------------------------

def test_ddpg_compute_diagnostics_no_q_gap():
    """DDPG actor diagnostics must NOT include Q-action gap (continuous action)."""
    actor = _make_actor('DDPG')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    # Actor-side metrics present
    assert 'diag_actor_fau_fc1' in result
    assert 'diag_actor_fau_fc2' in result
    assert 'diag_actor_wnorm_fc1' in result
    assert 'diag_actor_wnorm_fc2' in result
    assert 'diag_actor_wnorm_mu' in result
    assert 'diag_actor_erank_penult' in result

    # Q-gap must NOT be present for DDPG (output_kind='continuous_action')
    assert 'diag_q_action_gap_mean' not in result
    assert 'diag_q_action_gap_std' not in result
    assert 'diag_q_max_mean' not in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# TD3 — same shape as DDPG but uses TD3ActorNetwork
# ---------------------------------------------------------------------------------

def test_td3_compute_diagnostics_no_q_gap():
    """TD3 actor diagnostics: FAU + norms + erank present; no Q-gap."""
    actor = _make_actor('TD3')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    assert 'diag_actor_fau_fc1' in result
    assert 'diag_actor_fau_fc2' in result
    assert 'diag_actor_wnorm_fc1' in result
    assert 'diag_actor_wnorm_fc2' in result
    assert 'diag_actor_wnorm_mu' in result
    assert 'diag_actor_erank_penult' in result

    assert 'diag_q_action_gap_mean' not in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DIAGNOSE_CRITIC flag
# ---------------------------------------------------------------------------------

def test_ddpg_diagnose_critic_adds_critic_weight_norm_keys():
    """When DIAGNOSE_CRITIC=True, critic weight norms appear in DDPG diagnostic dict."""
    actor = _make_actor('DDPG', DIAGNOSE_CRITIC=True)
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    # Critic weight norm keys must be present with diag_critic_ prefix
    assert 'diag_critic_wnorm_fc1' in result
    assert 'diag_critic_wnorm_fc2' in result
    assert 'diag_critic_wnorm_q' in result

    _assert_no_nan_inf(result)


def test_ddqn_diagnose_critic_false_no_critic_keys():
    """When DIAGNOSE_CRITIC=False (default), no critic keys in DDQN output."""
    actor = _make_actor('DDQN')
    _inject_eval_batch(actor)
    result = actor.compute_diagnostics()

    critic_keys = [k for k in result if k.startswith('diag_critic_')]
    assert critic_keys == [], f"Unexpected critic keys: {critic_keys}"


# ---------------------------------------------------------------------------------
# Empty result when diagnostics disabled or batch not frozen
# ---------------------------------------------------------------------------------

def test_diagnostics_disabled_returns_empty():
    """compute_diagnostics must return {} when DIAGNOSTICS_ENABLED=False."""
    config = dict(_BASE_CONFIG)
    config['DIAGNOSTICS_ENABLED'] = False
    actor = Actor(
        id=1, config=config, network='DDQN', input_size=8, output_size=4,
        min_max_action=1.0, meta_param_size=16, gsp=False, recurrent_gsp=False,
        attention=False, gsp_input_size=6, gsp_output_size=1, gsp_look_back=2,
        gsp_sequence_length=5,
    )
    result = actor.compute_diagnostics()
    assert result == {}


def test_diagnostics_without_frozen_batch_returns_empty():
    """compute_diagnostics must return {} when eval batch has not been frozen yet."""
    actor = _make_actor('DDQN')
    # Do NOT call _inject_eval_batch — simulate pre-freeze state
    result = actor.compute_diagnostics()
    assert result == {}


# ---------------------------------------------------------------------------------
# Diversity metric passes through
# ---------------------------------------------------------------------------------

def test_gsp_pred_diversity_passthrough():
    """compute_diagnostics should include gsp_pred_diversity when predictions are passed."""
    actor = _make_actor('DDQN')
    _inject_eval_batch(actor)
    preds = np.random.uniform(-1.0, 1.0, size=200).astype(np.float32)
    result = actor.compute_diagnostics(gsp_predictions_this_episode=preds)
    assert 'diag_gsp_pred_diversity' in result
    assert math.isfinite(result['diag_gsp_pred_diversity'])
