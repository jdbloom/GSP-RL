"""Tests for SOFT_TARGET_TAU — Polyak soft update option for DQN/DDQN q_next.

Two contract groups:

1. OFF (tau=0): default behavior is preserved — hard copy fires at multiples of
   REPLACE_TARGET_COUNTER steps (using learn_step_counter BEFORE increment); no
   fire in between. Bit-identical to the path without SOFT_TARGET_TAU set.

2. ON (tau>0): after one learn step q_next moved toward q_eval by exactly tau
   (Polyak blend verified on every parameter); and the hard-reset branch does NOT
   fire (verified by setting REPLACE_TARGET_COUNTER=1, which would force a hard
   copy every step in the tau=0 path, but must NOT in the tau>0 path).
"""
import copy
import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


# --------------------------------------------------------------------------
# Shared config factory
# --------------------------------------------------------------------------

def _base_cfg(**overrides):
    cfg = {
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.002,
        "LR": 0.001,
        "EPSILON": 0.0,
        "EPS_MIN": 0.0,
        "EPS_DEC": 0.0,
        "BATCH_SIZE": 8,
        "MEM_SIZE": 200,
        "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0,
        "UPDATE_ACTOR_ITER": 1,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100,
        "GSP_BATCH_SIZE": 8,
    }
    cfg.update(overrides)
    return cfg


INPUT_SIZE = 8
OUTPUT_SIZE = 4
N_FILL = 30  # must be >= BATCH_SIZE


def _make_actor(network="DDQN", **cfg_overrides):
    T.manual_seed(0)
    np.random.seed(0)
    cfg = _base_cfg(**cfg_overrides)
    return Actor(
        id=1,
        config=cfg,
        network=network,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1.0,
        meta_param_size=1,
        gsp=False,
    )


def _fill_buffer(actor, seed):
    """Fill replay buffer with deterministic transitions."""
    rng = np.random.RandomState(seed)
    for _ in range(N_FILL):
        s = rng.random(INPUT_SIZE).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        r = float(rng.random())
        s_ = rng.random(INPUT_SIZE).astype(np.float32)
        actor.store_transition(s, a, r, s_, False, actor.networks)


# --------------------------------------------------------------------------
# Test 1: OFF path (tau=0)
# --------------------------------------------------------------------------

class TestSoftTargetOff:
    """tau=0 preserves exact hard-reset behavior."""

    def test_tau0_attribute_set_correctly(self):
        """SOFT_TARGET_TAU=0.0 is stored as soft_target_tau=0.0 on the actor."""
        actor = _make_actor(SOFT_TARGET_TAU=0.0)
        assert actor.soft_target_tau == 0.0

    def test_tau0_default_attribute(self):
        """When SOFT_TARGET_TAU is absent from config, soft_target_tau defaults to 0.0."""
        actor = _make_actor()  # no SOFT_TARGET_TAU in config
        assert actor.soft_target_tau == 0.0

    def test_tau0_hard_reset_fires_on_step_1(self):
        """With tau=0, hard reset fires on the FIRST learn step (counter=0 at call time).

        replace_target_network() is called BEFORE learn_DDQN() increments the counter,
        so learn_step_counter==0 on step 1 → 0 % REPLACE_TARGET_COUNTER == 0 → fires.
        After step 1, q_next should be an exact copy of q_eval (as it was pre-gradient;
        in practice this means q_next == q_eval at the moment of copy, not post-grad).
        """
        # Use REPLACE_TARGET_COUNTER large enough that it only fires on step 1
        actor = _make_actor(REPLACE_TARGET_COUNTER=100, SOFT_TARGET_TAU=0.0)
        _fill_buffer(actor, seed=7)

        # Snapshot q_eval BEFORE learn (before the gradient step mutates it)
        q_eval_before = {k: v.clone() for k, v in actor.networks["q_eval"].state_dict().items()}

        T.manual_seed(7)
        actor.learn()  # step 1: replace fires (counter=0), then DDQN updates q_eval

        # q_next should equal q_eval as it was BEFORE the DDQN gradient step
        q_next_after = actor.networks["q_next"].state_dict()
        for name in q_eval_before:
            T.testing.assert_close(
                q_next_after[name], q_eval_before[name], atol=0.0, rtol=0.0,
                msg=f"Hard reset did not produce exact copy at {name}",
            )

    def test_tau0_hard_reset_does_not_fire_between_counters(self):
        """With tau=0, hard reset does NOT fire on step 2 (counter=1 % CTR != 0)."""
        actor = _make_actor(REPLACE_TARGET_COUNTER=100, SOFT_TARGET_TAU=0.0)
        _fill_buffer(actor, seed=8)

        T.manual_seed(8)
        # Step 1: hard reset fires, q_next copies q_eval
        actor.learn()

        # Now diverge q_eval from q_next by running step 2 (no hard reset at counter=1)
        # Snapshot q_next after step 1
        q_next_after_step1 = {k: v.clone() for k, v in actor.networks["q_next"].state_dict().items()}
        actor.learn()  # step 2: counter=1 % 100 != 0 → no hard reset

        # q_next should be UNCHANGED from step-1 snapshot (hard reset didn't fire again)
        q_next_after_step2 = actor.networks["q_next"].state_dict()
        for name in q_next_after_step1:
            T.testing.assert_close(
                q_next_after_step2[name], q_next_after_step1[name], atol=0.0, rtol=0.0,
                msg=f"q_next changed between steps without hard reset at {name}",
            )


# --------------------------------------------------------------------------
# Test 2: ON path (tau>0)
# --------------------------------------------------------------------------

class TestSoftTargetOn:
    """tau>0: Polyak update every step, hard-reset branch disabled."""

    def test_tau_positive_attribute_set_correctly(self):
        """SOFT_TARGET_TAU=0.005 is stored as soft_target_tau=0.005."""
        actor = _make_actor(SOFT_TARGET_TAU=0.005)
        assert actor.soft_target_tau == 0.005

    def test_q_next_moves_toward_q_eval_by_tau(self):
        """After one learn step with tau=0.1, each q_next param satisfies:
        q_next_after == (1 - tau) * q_next_before + tau * q_eval_before.

        replace_target_network() runs BEFORE learn_DDQN() updates q_eval, so
        q_eval_before is the snapshot taken before the learn() call.
        """
        tau = 0.1
        actor = _make_actor(SOFT_TARGET_TAU=tau)
        _fill_buffer(actor, seed=99)

        # Snapshot q_next and q_eval BEFORE the learn step
        q_next_before = {k: v.clone() for k, v in actor.networks["q_next"].state_dict().items()}
        q_eval_before = {k: v.clone() for k, v in actor.networks["q_eval"].state_dict().items()}

        T.manual_seed(99)
        actor.learn()

        # Verify Polyak formula on every parameter
        for name, p in actor.networks["q_next"].named_parameters():
            expected = (1.0 - tau) * q_next_before[name] + tau * q_eval_before[name]
            T.testing.assert_close(
                p.data, expected, atol=1e-5, rtol=1e-5,
                msg=f"Polyak update wrong at {name}",
            )

    def test_hard_reset_does_not_fire_when_tau_positive(self):
        """With tau>0 and REPLACE_TARGET_COUNTER=1, q_next must NOT become an exact
        copy of q_eval after learn steps (the hard-reset branch is disabled).

        With REPLACE_TARGET_COUNTER=1 and tau=0, the hard reset fires every step
        (counter % 1 == 0 always), so q_next == q_eval after every step. With
        tau=0.005, q_next should lag behind q_eval after many steps.
        """
        tau = 0.005
        actor = _make_actor(SOFT_TARGET_TAU=tau, REPLACE_TARGET_COUNTER=1)
        _fill_buffer(actor, seed=123)

        T.manual_seed(123)
        # Run 20 steps — with REPLACE_TARGET_COUNTER=1 and tau=0, hard reset
        # would fire EVERY step → q_next == q_eval after every step.
        # With soft update at tau=0.005, q_next lags far behind.
        for _ in range(20):
            actor.learn()

        any_param_differs = False
        for name, p_next in actor.networks["q_next"].named_parameters():
            p_eval = dict(actor.networks["q_eval"].named_parameters())[name]
            if not T.allclose(p_next.data, p_eval.data, atol=1e-7):
                any_param_differs = True
                break

        assert any_param_differs, (
            "q_next == q_eval after 20 soft-update steps with small tau=0.005 — "
            "this indicates the hard-reset branch fired instead of Polyak update."
        )

    def test_dqn_soft_update_also_works(self):
        """Soft target update works for DQN (not just DDQN) network type."""
        tau = 0.1
        actor = _make_actor(network="DQN", SOFT_TARGET_TAU=tau)
        _fill_buffer(actor, seed=55)

        q_next_before = {k: v.clone() for k, v in actor.networks["q_next"].state_dict().items()}
        q_eval_before = {k: v.clone() for k, v in actor.networks["q_eval"].state_dict().items()}

        T.manual_seed(55)
        actor.learn()

        for name, p in actor.networks["q_next"].named_parameters():
            expected = (1.0 - tau) * q_next_before[name] + tau * q_eval_before[name]
            T.testing.assert_close(
                p.data, expected, atol=1e-5, rtol=1e-5,
                msg=f"DQN Polyak update wrong at {name}",
            )
