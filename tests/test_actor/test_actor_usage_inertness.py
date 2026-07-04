"""Golden functional-inertness guard for the actor-usage diagnostics (M1/M3).

The M1 saliency metric runs a backward pass to obtain dQ/d(input). If that
backward shared tensors or gradient buffers with the real training path, it
would pollute the optimizer's gradients or parameters. This test proves it does
not: it runs learn + compute_diagnostics steps on a GSP-enabled actor (so the
new metrics actually fire) and asserts that both network parameters AND the
Adam optimizer state (exp_avg / exp_avg_sq / step) are BIT-IDENTICAL to a
control run where diagnostics — and thus the metrics — are never computed.

This mirrors the Tier-2 golden pattern in
test_tier2_grad_zero_functional_inertness.py, extended to cover optimizer state
because saliency calls backward (unlike the pure-forward grad_zero metric).
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
_CONFIG_PATH = os.path.join(_HERE, "config.yml")

with open(_CONFIG_PATH, "r") as _f:
    _BASE_CONFIG = yaml.safe_load(_f)


INPUT_SIZE = 8
OUTPUT_SIZE = 4
PRED_WIDTH = 3


def _make_gsp_actor(seed: int, diagnostics_enabled: bool) -> Actor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(_BASE_CONFIG)
    cfg["DIAGNOSTICS_ENABLED"] = diagnostics_enabled
    cfg["DIAGNOSTICS_BATCH_SIZE"] = 32
    cfg["DIAGNOSTICS_FREEZE_EPISODE"] = 0
    cfg["DIAGNOSE_CRITIC"] = False
    return Actor(
        id=1,
        config=cfg,
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


def _fill_replay(actor: Actor, n: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.standard_normal(actor.network_input_size).astype(np.float32)
        s_ = rng.standard_normal(actor.network_input_size).astype(np.float32)
        a = int(rng.integers(0, OUTPUT_SIZE))
        r = float(rng.standard_normal())
        actor.networks["replay"].store_transition(s, a, r, s_, False)


def _train_steps(actor: Actor, n_steps: int):
    for _ in range(n_steps):
        actor.learn_DDQN(networks=actor.networks)


def _q_net(actor: Actor):
    return actor._main_network(actor.networks)


def _weights_snapshot(actor: Actor) -> dict:
    snap = {}
    for name, param in actor.networks.items():
        if hasattr(param, "state_dict"):
            for pname, pval in param.state_dict().items():
                snap[f"{name}.{pname}"] = pval.clone().detach()
    return snap


def _optimizer_snapshot(actor: Actor) -> dict:
    """Flatten the Adam optimizer state (exp_avg, exp_avg_sq, step) of the
    q_eval network into a comparable dict of tensors."""
    net = _q_net(actor)
    opt = net.optimizer
    snap = {}
    for gi, group in enumerate(opt.param_groups):
        for pi, p in enumerate(group["params"]):
            st = opt.state.get(p, {})
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    snap[f"g{gi}.p{pi}.{k}"] = v.clone().detach()
                else:
                    snap[f"g{gi}.p{pi}.{k}"] = torch.tensor(float(v))
    return snap


def _assert_equal(a: dict, b: dict, label: str):
    assert set(a.keys()) == set(b.keys()), f"[{label}] keys differ: {set(a) ^ set(b)}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"[{label}] {k} diverged"


def test_actor_usage_metrics_do_not_perturb_training():
    """Computing M1 saliency (backward) + M3 wnorm must leave the real
    optimizer's params and Adam state bit-identical to a no-diagnostics control."""
    SEED = 7
    N_FILL = 64
    N_STEPS = 10

    # Control: diagnostics OFF — the new metrics are never computed.
    actor_off = _make_gsp_actor(seed=SEED, diagnostics_enabled=False)
    _fill_replay(actor_off, n=N_FILL, seed=SEED)
    _train_steps(actor_off, n_steps=N_STEPS)
    w_off = _weights_snapshot(actor_off)
    o_off = _optimizer_snapshot(actor_off)

    # Treatment: diagnostics ON — compute_diagnostics (with M1/M3 active) runs
    # after every training step.
    actor_on = _make_gsp_actor(seed=SEED, diagnostics_enabled=True)
    _fill_replay(actor_on, n=N_FILL, seed=SEED)
    actor_on.freeze_diagnostic_batch()
    for _ in range(N_STEPS):
        _train_steps(actor_on, n_steps=1)
        result = actor_on.compute_diagnostics()
        # Sanity: the new metrics really did fire on this GSP-enabled actor.
        assert "diag_gsp_actor_saliency" in result
        assert "diag_gsp_actor_wnorm_pred_rel" in result
    w_on = _weights_snapshot(actor_on)
    o_on = _optimizer_snapshot(actor_on)

    _assert_equal(w_off, w_on, "params")
    _assert_equal(o_off, o_on, "optimizer state")


def test_saliency_backward_leaves_no_grad_on_qnet():
    """After compute_diagnostics, the q_eval params carry no lingering .grad
    from the saliency backward (set_to_none clears them)."""
    actor = _make_gsp_actor(seed=3, diagnostics_enabled=True)
    _fill_replay(actor, n=64, seed=3)
    actor.freeze_diagnostic_batch()
    _train_steps(actor, n_steps=1)
    _ = actor.compute_diagnostics()
    net = _q_net(actor)
    for name, p in net.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, (
            f"lingering grad on {name} after saliency diagnostic"
        )


def test_grad_enabled_state_restored():
    """compute_diagnostics must restore the ambient grad-enabled state even
    though saliency forces torch.enable_grad() internally."""
    actor = _make_gsp_actor(seed=5, diagnostics_enabled=True)
    _fill_replay(actor, n=64, seed=5)
    actor.freeze_diagnostic_batch()
    _train_steps(actor, n_steps=1)
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        _ = actor.compute_diagnostics()
        assert not torch.is_grad_enabled(), "grad-enabled state leaked out of diagnostics"
