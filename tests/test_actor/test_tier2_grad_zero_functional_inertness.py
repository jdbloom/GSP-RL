"""Functional-inertness test for the Tier 2 auto-merge gate.

Asserts that enabling compute_grad_zero_fraction via the DIAGNOSTIC_PROFILE's
grad_layers key does NOT change training behavior. Run training for N learn
steps with and without the diagnostic active at the same seed; assert all
network weights and optimizer state are byte-identical.

This is the canonical Tier 2 auto-merge gate: if a new diagnostic passes
this test (plus the shape test), it auto-merges without approval.

See docs/diagnostics/proposed.md § 'Auto-merge criteria for Tier 2'.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch
import yaml

from gsp_rl.src.actors import Actor


_HERE = os.path.dirname(os.path.realpath(__file__))
_CONFIG_PATH = os.path.join(_HERE, 'config.yml')

with open(_CONFIG_PATH, 'r') as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


def _make_actor(network: str, seed: int, diagnostics_enabled: bool) -> Actor:
    """Build actor with fixed seed + configurable diagnostics flag."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(_BASE_CONFIG)
    cfg['DIAGNOSTICS_ENABLED'] = diagnostics_enabled
    cfg['DIAGNOSTICS_BATCH_SIZE'] = 32
    cfg['DIAGNOSTICS_FREEZE_EPISODE'] = 0
    cfg['DIAGNOSE_CRITIC'] = False
    return Actor(
        id=1,
        config=cfg,
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
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def _fill_replay(actor: Actor, n: int, seed: int):
    """Push n synthetic transitions into actor's replay buffer, deterministically."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.standard_normal(actor.network_input_size).astype(np.float32)
        s_ = rng.standard_normal(actor.network_input_size).astype(np.float32)
        a = rng.standard_normal(actor.output_size).astype(np.float32) if actor.networks.get('learning_scheme') in ('DDPG', 'TD3') else rng.integers(0, 4)
        r = float(rng.standard_normal())
        d = False
        actor.networks['replay'].store_transition(s, a, r, s_, d)


def _train_steps(actor: Actor, n_steps: int, batch_size: int = 32):
    """Invoke the appropriate learn_* method n_steps times.

    GSP-RL learn methods take (networks, ...) — batch_size is hard-coded from
    config at actor init. We call them directly here.
    """
    scheme = actor.networks.get('learning_scheme', '')
    # Ensure batch_size is small enough relative to mem_ctr; already filled 64 above.
    for _ in range(n_steps):
        if scheme == 'DDPG':
            actor.learn_DDPG(networks=actor.networks)
        elif scheme == 'DDQN':
            actor.learn_DDQN(networks=actor.networks)
        elif scheme == 'DQN':
            actor.learn_DQN(networks=actor.networks)


def _weights_snapshot(actor: Actor) -> dict:
    """Flatten all network parameter tensors into a single comparable dict."""
    snap = {}
    for name, param in actor.networks.items():
        if hasattr(param, 'state_dict'):
            for pname, pval in param.state_dict().items():
                snap[f"{name}.{pname}"] = pval.clone().detach()
    return snap


def _assert_weights_equal(a: dict, b: dict, label: str):
    """Byte-identical comparison across snapshots."""
    assert set(a.keys()) == set(b.keys()), f"[{label}] keys differ: {set(a) ^ set(b)}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"[{label}] {k} diverged"


@pytest.mark.parametrize("scheme", ["DDPG", "DDQN", "DQN"])
def test_grad_zero_profile_is_functionally_inert(scheme):
    """Enabling grad_layers in DIAGNOSTIC_PROFILE must not change training.

    Builds two actors with the same seed. On one, we compute diagnostics
    (which now includes grad_zero_fraction due to the profile update).
    On the other, we don't. After N training steps, assert weights byte-equal.
    """
    SEED = 42
    N_FILL = 64
    N_STEPS = 10
    BATCH_SIZE = 32

    # Control: diagnostics OFF, no compute_diagnostics calls.
    actor_off = _make_actor(scheme, seed=SEED, diagnostics_enabled=False)
    _fill_replay(actor_off, n=N_FILL, seed=SEED)
    _train_steps(actor_off, n_steps=N_STEPS, batch_size=BATCH_SIZE)
    snap_off = _weights_snapshot(actor_off)

    # Treatment: diagnostics ON, compute_diagnostics called between training steps.
    # If compute_grad_zero_fraction (newly enabled via grad_layers) perturbs training,
    # snap_on will diverge from snap_off.
    actor_on = _make_actor(scheme, seed=SEED, diagnostics_enabled=True)
    _fill_replay(actor_on, n=N_FILL, seed=SEED)
    # Freeze the eval batch so compute_diagnostics has something to work with.
    actor_on.freeze_diagnostic_batch()
    for _ in range(N_STEPS):
        _train_steps(actor_on, n_steps=1, batch_size=BATCH_SIZE)
        _ = actor_on.compute_diagnostics()
    snap_on = _weights_snapshot(actor_on)

    _assert_weights_equal(snap_off, snap_on, f"{scheme} grad_layers profile")


@pytest.mark.parametrize("scheme", ["DDPG", "DDQN", "DQN"])
def test_grad_zero_outputs_expected_shape(scheme):
    """Enabling grad_layers must produce expected diag_*_grad_zero_* keys."""
    actor = _make_actor(scheme, seed=42, diagnostics_enabled=True)
    _fill_replay(actor, n=64, seed=42)
    actor.freeze_diagnostic_batch()
    _train_steps(actor, n_steps=1, batch_size=32)
    out = actor.compute_diagnostics()

    # Expect at least one diag_*_grad_zero_* key per declared grad_layer.
    grad_zero_keys = [k for k in out.keys() if 'grad_zero' in k]
    assert len(grad_zero_keys) > 0, f"{scheme}: no grad_zero keys produced. Got: {sorted(out.keys())}"
    for k in grad_zero_keys:
        v = out[k]
        assert np.isfinite(v), f"{scheme}: {k}={v} not finite"
        assert 0.0 <= v <= 1.0, f"{scheme}: {k}={v} out of [0,1]"
