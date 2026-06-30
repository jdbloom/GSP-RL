"""T2 golden-equivalence gate for the sample_memory + buffers inert optimization.

T2 targets (all bit-exact, no precision change):
- sample_memory and per-scheme sample blocks: replace T.tensor(numpy).to(device)
  double-copy with T.as_tensor/from_numpy + single .to(device, non_blocking=True).
- ReplayBuffer.store_transition: cache a zero gsp_obs vector instead of
  allocating np.zeros(...) per None-store.

Each test loads a frozen baseline reference (loss + post-step params), recomputes
the same deterministic learn step on the current code, and asserts exact-to-
tolerance equivalence.  The change must be purely inert — no numeric change.

float64->float32 STORAGE change for sequence/attention buffers is explicitly NOT
tested here; that sub-bundle is deferred (not bit-exact, needs integration cell).
"""
import os
import sys

import numpy as np
import pytest

# Allow ``python -m pytest tests/test_buffers/test_golden_t2.py`` from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEARN_AIDS_TESTS = os.path.join(os.path.dirname(_HERE), "test_learning_aids")
if _LEARN_AIDS_TESTS not in sys.path:
    sys.path.insert(0, _LEARN_AIDS_TESTS)

_REFS_DIR = os.path.join(os.path.dirname(_HERE), "test_learning_aids", "golden_refs")

_RTOL = 1e-6
_ATOL = 1e-6


def _assert_matches(got, ref):
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=_RTOL, atol=_ATOL,
                               err_msg="loss diverged from frozen baseline")
    np.testing.assert_allclose(got["params"], ref["params"], rtol=_RTOL, atol=_ATOL,
                               err_msg="params diverged from frozen baseline")


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def test_golden_t2_dqn():
    from golden_helpers import one_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t2_dqn.npz"))

    got = one_learn_step("DQN")

    # Determinism: two in-process calls must be bit-identical.
    got2 = one_learn_step("DQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DQN learn step is not deterministic"

    _assert_matches(got, ref)


# ---------------------------------------------------------------------------
# DDQN
# ---------------------------------------------------------------------------

def test_golden_t2_ddqn():
    from golden_helpers import one_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t2_ddqn.npz"))

    got = one_learn_step("DDQN")

    got2 = one_learn_step("DDQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DDQN learn step is not deterministic"

    _assert_matches(got, ref)


# ---------------------------------------------------------------------------
# DDPG
# ---------------------------------------------------------------------------

def test_golden_t2_ddpg():
    from golden_helpers import one_learn_step_continuous
    ref = np.load(os.path.join(_REFS_DIR, "t2_ddpg.npz"))

    got = one_learn_step_continuous("DDPG")

    got2 = one_learn_step_continuous("DDPG")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DDPG learn step is not deterministic"

    _assert_matches(got, ref)


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------

def test_golden_t2_td3():
    from golden_helpers import one_learn_step_continuous
    ref = np.load(os.path.join(_REFS_DIR, "t2_td3.npz"))

    got = one_learn_step_continuous("TD3")

    got2 = one_learn_step_continuous("TD3")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "TD3 learn step is not deterministic"

    _assert_matches(got, ref)
