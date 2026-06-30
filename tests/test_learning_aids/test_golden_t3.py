"""T3 golden-equivalence gate for the _check_nan combine optimization.

T3 target (partial — ONLY _check_nan combine shipped):
- Combine the two-branch isinstance+isnan+isinf check into a single
  (isnan|isinf).any() call (same logical result, one fewer GPU sync).

The diagnostic-gating sub-bundle (gating per-step GPU→CPU syncs in
learn_DDQN_e2e return dict, learn_gsp_jepa latent-SVD, learn_gsp_mse
batch-corr, and E2E grad-norm comps behind diagnostics_cadence) is NOT
shipped — the learn-step return dicts are always consumed by callers and
gating them would silently drop data downstream analysis depends on.  That
sub-bundle needs principal attention.

Each test loads a frozen baseline reference (loss + post-step params),
recomputes the same deterministic learn step on the current code, and asserts
exact-to-tolerance equivalence.  _check_nan must not change any computed value.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REFS_DIR = os.path.join(_HERE, "golden_refs")

_RTOL = 1e-6
_ATOL = 1e-6


def _assert_matches(got, ref):
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=_RTOL, atol=_ATOL,
                               err_msg="loss diverged from frozen baseline")
    np.testing.assert_allclose(got["params"], ref["params"], rtol=_RTOL, atol=_ATOL,
                               err_msg="params diverged from frozen baseline")


def test_golden_t3_dqn():
    from golden_helpers import one_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t3_dqn.npz"))

    got = one_learn_step("DQN")

    got2 = one_learn_step("DQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DQN learn step is not deterministic"

    _assert_matches(got, ref)


def test_golden_t3_ddqn():
    from golden_helpers import one_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t3_ddqn.npz"))

    got = one_learn_step("DDQN")

    got2 = one_learn_step("DDQN")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DDQN learn step is not deterministic"

    _assert_matches(got, ref)


def test_golden_t3_ddpg():
    from golden_helpers import one_learn_step_continuous
    ref = np.load(os.path.join(_REFS_DIR, "t3_ddpg.npz"))

    got = one_learn_step_continuous("DDPG")

    got2 = one_learn_step_continuous("DDPG")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "DDPG learn step is not deterministic"

    _assert_matches(got, ref)


def test_golden_t3_td3():
    from golden_helpers import one_learn_step_continuous
    ref = np.load(os.path.join(_REFS_DIR, "t3_td3.npz"))

    got = one_learn_step_continuous("TD3")

    got2 = one_learn_step_continuous("TD3")
    np.testing.assert_array_equal(got["params"], got2["params"])
    assert got["loss"] == got2["loss"], "TD3 learn step is not deterministic"

    _assert_matches(got, ref)
