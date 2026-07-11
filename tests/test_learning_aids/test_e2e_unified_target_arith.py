"""GSP_E2E_UNIFIED_TARGET_ARITH flag-ON semantics.

The flag-OFF (default) path is gated by test_golden_e2e_arith.py. Here:

  1. flag parse + defaults;
  2. the fail-loud startup line (ENGAGED / off, only on E2E runs);
  3. flag ON at NEUTRAL stabilizer values == legacy (parity sanity);
  4. flag ON: the E2E Q-target equals the _q_target formula
     (reward_scale * rewards + gamma * bootstrap) computed BY HAND on the
     exact sampled batch with the pre-learn networks;
  5. Q_TARGET_CLIP engages at the boundary (hand-computed clamped target
     matches; the unclamped target does not);
  6. the critic grad clip mirrors learn_DDQN (clipped when ON, NOT clipped
     when OFF, for the same GRAD_CLIP_NORM);
  7. the E2E head-loss path (lambda * gsp_mse_loss) is bit-identical under
     the flag.
"""
import copy
import logging

import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.learning_aids import NetworkAids

from e2e_arith_helpers import (
    BATCH_SIZE,
    E2E_BASE_CONFIG,
    ENV_OBS_SIZE,
    make_e2e_setup,
    one_e2e_learn_step,
)

_GAMMA = E2E_BASE_CONFIG["GAMMA"]


# --- 1. flag parse ----------------------------------------------------------

def test_flag_default_false():
    aids = NetworkAids(dict(E2E_BASE_CONFIG))
    assert aids.gsp_e2e_unified_target_arith is False


def test_flag_parses_true():
    cfg = dict(E2E_BASE_CONFIG)
    cfg["GSP_E2E_UNIFIED_TARGET_ARITH"] = True
    aids = NetworkAids(cfg)
    assert aids.gsp_e2e_unified_target_arith is True


def test_engaged_attr_true_when_e2e_and_flag():
    """gsp_e2e_unified_arith_engaged is the single condition source RL-CT
    Main.py keys its per-run python.log startup line on (the stelaris.learn
    lines below never reach a handler in production)."""
    cfg = dict(E2E_BASE_CONFIG)
    cfg["GSP_E2E_UNIFIED_TARGET_ARITH"] = True
    aids = NetworkAids(cfg)
    assert aids.gsp_e2e_unified_arith_engaged is True


def test_engaged_attr_false_by_default():
    aids = NetworkAids(dict(E2E_BASE_CONFIG))  # E2E on, flag absent
    assert aids.gsp_e2e_unified_arith_engaged is False


def test_engaged_attr_false_without_e2e():
    """The flag governs only learn_DDQN_e2e; on a non-E2E arm the attribute
    must read False even with the raw flag set, so the caller's ENGAGED line
    can never claim unified arithmetic on an IC arm."""
    cfg = dict(E2E_BASE_CONFIG)
    cfg["GSP_E2E_ENABLED"] = False
    cfg["GSP_E2E_UNIFIED_TARGET_ARITH"] = True
    aids = NetworkAids(cfg)
    assert aids.gsp_e2e_unified_arith_engaged is False


# --- 2. fail-loud startup line ----------------------------------------------

def test_startup_line_engaged(caplog):
    cfg = dict(E2E_BASE_CONFIG)
    cfg.update({
        "GSP_E2E_UNIFIED_TARGET_ARITH": True,
        "REWARD_SCALE": 0.1,
        "Q_TARGET_CLIP": 1000.0,
        "GRAD_CLIP_NORM": 10.0,
    })
    with caplog.at_level(logging.INFO, logger="stelaris.learn"):
        NetworkAids(cfg)
    msgs = [r.getMessage() for r in caplog.records]
    engaged = [m for m in msgs if "GSP_E2E_UNIFIED_TARGET_ARITH: ENGAGED" in m]
    assert len(engaged) == 1, msgs
    # The line must state the arithmetic as consumed (fail-loud contract).
    assert "reward_scale(0.1)" in engaged[0]
    assert "Q_TARGET_CLIP=1000.0" in engaged[0]
    assert "critic grad clip=10.0" in engaged[0]


def test_startup_line_off(caplog):
    with caplog.at_level(logging.INFO, logger="stelaris.learn"):
        NetworkAids(dict(E2E_BASE_CONFIG))  # E2E on, flag absent
    msgs = [r.getMessage() for r in caplog.records]
    assert any("GSP_E2E_UNIFIED_TARGET_ARITH: off" in m for m in msgs), msgs
    assert not any("GSP_E2E_UNIFIED_TARGET_ARITH: ENGAGED" in m for m in msgs)


def test_no_startup_line_without_e2e(caplog):
    """The flag governs only the E2E learn fn; a non-E2E run must not emit
    the line (it would be a misleading arithmetic claim on IC arms)."""
    cfg = dict(E2E_BASE_CONFIG)
    cfg["GSP_E2E_ENABLED"] = False
    cfg["GSP_E2E_UNIFIED_TARGET_ARITH"] = True
    with caplog.at_level(logging.INFO, logger="stelaris.learn"):
        NetworkAids(cfg)
    assert not any(
        "GSP_E2E_UNIFIED_TARGET_ARITH" in r.getMessage() for r in caplog.records
    )


# --- 3. neutral-stabilizer parity sanity -------------------------------------

def test_flag_on_neutral_stabilizers_identical_to_legacy():
    """With reward_scale=1, clip=0, grad_clip=0 the unified arithmetic is
    algebraically the legacy arithmetic — results must match bit-for-bit."""
    off = one_e2e_learn_step()
    on = one_e2e_learn_step({"GSP_E2E_UNIFIED_TARGET_ARITH": True})
    assert on["ddqn_loss"] == off["ddqn_loss"]
    assert on["gsp_mse_loss"] == off["gsp_mse_loss"]
    np.testing.assert_array_equal(on["q_eval_params"], off["q_eval_params"])
    np.testing.assert_array_equal(on["gsp_actor_params"], off["gsp_actor_params"])


# --- 4/5. hand-computed target ------------------------------------------------

def _replicate_e2e_forward(aids, networks_pre, gsp_pre):
    """Re-sample the SAME batch (np RNG re-pinned by the caller) and rebuild
    q_pred / bootstrap with the PRE-learn network clones, mirroring
    learn_DDQN_e2e's tensor flow exactly (scalar-slot scale, splice, DDQN
    double-Q bootstrap). Returns (rewards, q_pred, bootstrap, q_eval_clone)."""
    device = networks_pre["q_eval"].device
    result = networks_pre["replay"].sample_buffer(BATCH_SIZE)
    states_np, actions_np, rewards_np, states_np_, dones_np, gsp_obs_np, _ = result

    states = T.as_tensor(states_np, dtype=T.float32).to(device)
    actions = T.as_tensor(np.asarray(actions_np, dtype=np.float32)).to(device)
    rewards = T.as_tensor(rewards_np, dtype=T.float32).to(device)
    states_ = T.as_tensor(states_np_, dtype=T.float32).to(device)
    dones = T.as_tensor(dones_np).to(device)
    gsp_obs = T.as_tensor(gsp_obs_np, dtype=T.float32).to(device)

    with T.no_grad():
        gsp_pred = gsp_pre["actor"].forward(gsp_obs)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)
        # Scalar slot (K=1): the degrees(x/10)/x acting-splice scale.
        gsp_pred_actor = gsp_pred * float(np.degrees(1.0) / 10.0)
        augmented = T.cat(
            [states[:, :ENV_OBS_SIZE], gsp_pred_actor, states[:, ENV_OBS_SIZE + 1:]],
            dim=1,
        )
        indices = T.LongTensor(np.arange(BATCH_SIZE).astype(np.int64))
        q_pred = networks_pre["q_eval"](augmented)[indices, actions.type(T.LongTensor)]
        q_next = networks_pre["q_next"](states_)
        q_eval_next = networks_pre["q_eval"](states_)
        max_actions = T.argmax(q_eval_next, dim=1)
        q_next[dones] = 0.0
        bootstrap = q_next[indices, max_actions]
    return rewards, q_pred, bootstrap, networks_pre["q_eval"]


def test_flag_on_target_equals_q_target_formula_by_hand():
    """Flag ON with REWARD_SCALE=0.25 (no clip): the returned ddqn_loss must
    equal loss(0.25 * rewards + gamma * bootstrap, q_pred) computed BY HAND
    on the same batch with the pre-learn networks."""
    overrides = {
        "GSP_E2E_UNIFIED_TARGET_ARITH": True,
        "REWARD_SCALE": 0.25,
    }
    np.random.seed(0)
    T.manual_seed(0)
    aids, networks, gsp_networks = make_e2e_setup(overrides)
    networks_pre = {
        "q_eval": copy.deepcopy(networks["q_eval"]),
        "q_next": copy.deepcopy(networks["q_next"]),
        "replay": networks["replay"],  # learn does not mutate the buffer
    }
    gsp_pre = {"actor": copy.deepcopy(gsp_networks["actor"])}

    np.random.seed(123)
    T.manual_seed(123)
    diag = aids.learn_DDQN_e2e(networks, gsp_networks)

    # Re-pin the SAME np state the learn call sampled under.
    np.random.seed(123)
    T.manual_seed(123)
    rewards, q_pred, bootstrap, q_eval_pre = _replicate_e2e_forward(
        aids, networks_pre, gsp_pre)

    expected_target = 0.25 * rewards + _GAMMA * bootstrap  # the _q_target formula
    expected_loss = q_eval_pre.loss(expected_target, q_pred).item()
    np.testing.assert_allclose(diag["ddqn_loss"], expected_loss,
                               rtol=1e-6, atol=1e-6)

    # And it must NOT equal the legacy unscaled target's loss (the batch has
    # non-trivial rewards, so the two formulas genuinely differ here).
    legacy_loss = q_eval_pre.loss(rewards + _GAMMA * bootstrap, q_pred).item()
    assert abs(diag["ddqn_loss"] - legacy_loss) > 1e-8


def test_flag_on_clip_engages_at_boundary():
    """Flag ON, REWARD_SCALE=1, Q_TARGET_CLIP=5, rewards filled ~N(0,200):
    the hand-computed UNCLAMPED target must exceed the clip somewhere, and
    the returned loss must match the CLAMPED target (and not the unclamped)."""
    clip = 5.0
    overrides = {
        "GSP_E2E_UNIFIED_TARGET_ARITH": True,
        "REWARD_SCALE": 1.0,
        "Q_TARGET_CLIP": clip,
    }
    np.random.seed(0)
    T.manual_seed(0)
    aids, networks, gsp_networks = make_e2e_setup(
        overrides, reward_scale_fill=200.0)
    networks_pre = {
        "q_eval": copy.deepcopy(networks["q_eval"]),
        "q_next": copy.deepcopy(networks["q_next"]),
        "replay": networks["replay"],
    }
    gsp_pre = {"actor": copy.deepcopy(gsp_networks["actor"])}

    np.random.seed(123)
    T.manual_seed(123)
    diag = aids.learn_DDQN_e2e(networks, gsp_networks)

    np.random.seed(123)
    T.manual_seed(123)
    rewards, q_pred, bootstrap, q_eval_pre = _replicate_e2e_forward(
        aids, networks_pre, gsp_pre)

    unclamped = 1.0 * rewards + _GAMMA * bootstrap
    assert unclamped.abs().max().item() > clip, (
        "fixture failed to push the target past the clip boundary")
    clamped = T.clamp(unclamped, -clip, clip)
    assert clamped.abs().max().item() <= clip

    expected_loss = q_eval_pre.loss(clamped, q_pred).item()
    np.testing.assert_allclose(diag["ddqn_loss"], expected_loss,
                               rtol=1e-6, atol=1e-6)
    unclamped_loss = q_eval_pre.loss(unclamped, q_pred).item()
    assert abs(diag["ddqn_loss"] - unclamped_loss) > 1e-6


# --- 6. critic grad clip mirrors learn_DDQN -----------------------------------

def _qeval_grad_norm_after_learn(flag_on: bool) -> float:
    np.random.seed(0)
    T.manual_seed(0)
    aids, networks, gsp_networks = make_e2e_setup({
        "GSP_E2E_UNIFIED_TARGET_ARITH": flag_on,
        "GRAD_CLIP_NORM": 1e-3,
    })
    np.random.seed(123)
    T.manual_seed(123)
    aids.learn_DDQN_e2e(networks, gsp_networks)
    grads = [p.grad for p in networks["q_eval"].parameters() if p.grad is not None]
    assert grads, "q_eval received no gradients"
    return float(T.norm(T.stack([g.norm() for g in grads])).item())


def test_flag_on_clips_qeval_grad_like_learn_ddqn():
    assert _qeval_grad_norm_after_learn(flag_on=True) <= 1e-3 + 1e-6


def test_flag_off_does_not_clip_qeval_grad():
    """Same GRAD_CLIP_NORM, flag off: legacy behavior never clips q_eval in
    the E2E fn — the norm must exceed the clip value by a wide margin."""
    assert _qeval_grad_norm_after_learn(flag_on=False) > 1e-2


# --- 7. head-loss path unaffected ----------------------------------------------

def test_head_loss_path_bit_identical_under_flag():
    """lambda * gsp_mse_loss must be untouched: same seeds, flag off vs ON
    with aggressive stabilizers — gsp_mse_loss is bit-identical (the head's
    supervised pair (pred, label) does not depend on the TD target)."""
    off = one_e2e_learn_step({
        "REWARD_SCALE": 0.1, "Q_TARGET_CLIP": 5.0, "GRAD_CLIP_NORM": 1e-3,
    })
    on = one_e2e_learn_step({
        "GSP_E2E_UNIFIED_TARGET_ARITH": True,
        "REWARD_SCALE": 0.1, "Q_TARGET_CLIP": 5.0, "GRAD_CLIP_NORM": 1e-3,
    })
    assert on["gsp_mse_loss"] == off["gsp_mse_loss"]
    # ...while the TD side genuinely changed (guards against a vacuous pass).
    assert on["ddqn_loss"] != off["ddqn_loss"]
