"""T1 golden-equivalence gate for the DQN/DDQN learn-step inert optimization.

Each test loads a frozen baseline reference (loss + post-step q_eval params),
recomputes the same deterministic learn step on the current code, and asserts
exact-to-tolerance equivalence. A later behavior-preserving refactor of
``learn_DQN`` / ``learn_DDQN`` / ``_q_target`` / Actor ``__init__`` must keep
these green.

A determinism sub-assertion guards the fixture itself: two in-process calls must
match bit-for-tolerance. A non-deterministic fixture cannot gate anything.
"""
import os

import numpy as np

from golden_helpers import one_learn_step

_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_refs")

_RTOL = 1e-6
_ATOL = 1e-6


def _assert_matches(got, ref):
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(got["params"], ref["params"], rtol=_RTOL, atol=_ATOL)


def test_golden_dqn():
    ref = np.load(os.path.join(_REFS_DIR, "t1_dqn.npz"))

    got = one_learn_step("DQN")

    # Determinism: a second in-process call must reproduce the first exactly.
    got2 = one_learn_step("DQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"]

    _assert_matches(got, ref)


def test_golden_ddqn():
    ref = np.load(os.path.join(_REFS_DIR, "t1_ddqn.npz"))

    got = one_learn_step("DDQN")

    # Determinism: a second in-process call must reproduce the first exactly.
    got2 = one_learn_step("DDQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"]

    _assert_matches(got, ref)
