"""Golden gate for GSP_E2E_UNIFIED_TARGET_ARITH flag-OFF (the default).

References were frozen by ``freeze_e2e_arith.py`` at the merge-base of
fix/e2e-target-arith-parity (main BEFORE the flag existed). The default and
the explicit-False configurations must reproduce the pre-change learn step
exactly-to-tolerance for BOTH learn fns:

  * learn_DDQN_e2e          — the fn the flag modifies when ON;
  * learn_DDQN_jepa_coupled — verified bypass-free (already on _q_target);
                              frozen as insurance the flag work never
                              perturbs it.

Same tolerance + in-process determinism sub-assertion pattern as
``test_golden_t1.py``.
"""
import os

import numpy as np

from e2e_arith_helpers import one_e2e_learn_step, one_jepa_coupled_learn_step

_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_refs")

_RTOL = 1e-6
_ATOL = 1e-6


def _assert_matches(got, ref, scalar_keys, param_keys):
    for k in scalar_keys:
        np.testing.assert_allclose(got[k], ref[k], rtol=_RTOL, atol=_ATOL)
    for k in param_keys:
        np.testing.assert_allclose(got[k], ref[k], rtol=_RTOL, atol=_ATOL)


def test_golden_ddqn_e2e_flag_off_default():
    ref = np.load(os.path.join(_REFS_DIR, "e2e_arith_ddqn_e2e.npz"))

    got = one_e2e_learn_step()

    # Determinism: a second in-process call must reproduce the first exactly.
    got2 = one_e2e_learn_step()
    np.testing.assert_array_equal(got["q_eval_params"], got2["q_eval_params"])
    np.testing.assert_array_equal(got["gsp_actor_params"], got2["gsp_actor_params"])
    assert got["ddqn_loss"] == got2["ddqn_loss"]

    _assert_matches(
        got, ref,
        scalar_keys=("ddqn_loss", "gsp_mse_loss", "total_loss"),
        param_keys=("q_eval_params", "gsp_actor_params"),
    )


def test_golden_ddqn_e2e_flag_explicit_false():
    """An explicit False must be indistinguishable from an absent key."""
    ref = np.load(os.path.join(_REFS_DIR, "e2e_arith_ddqn_e2e.npz"))
    got = one_e2e_learn_step({"GSP_E2E_UNIFIED_TARGET_ARITH": False})
    _assert_matches(
        got, ref,
        scalar_keys=("ddqn_loss", "gsp_mse_loss", "total_loss"),
        param_keys=("q_eval_params", "gsp_actor_params"),
    )


def test_golden_ddqn_e2e_flag_off_ignores_stabilizer_keys():
    """Flag off, the E2E path must STAY legacy even when the stabilizer keys
    are set (that is the asymmetry the flag exists to close — off means
    'reproduce history', not 'partially unified')."""
    ref = np.load(os.path.join(_REFS_DIR, "e2e_arith_ddqn_e2e.npz"))
    got = one_e2e_learn_step({
        "GSP_E2E_UNIFIED_TARGET_ARITH": False,
        "REWARD_SCALE": 0.1,
        "Q_TARGET_CLIP": 1000.0,
        "GRAD_CLIP_NORM": 10.0,
    })
    _assert_matches(
        got, ref,
        scalar_keys=("ddqn_loss", "gsp_mse_loss", "total_loss"),
        param_keys=("q_eval_params", "gsp_actor_params"),
    )


def test_golden_jepa_coupled_flag_off_default():
    ref = np.load(os.path.join(_REFS_DIR, "e2e_arith_jepa_coupled.npz"))

    got = one_jepa_coupled_learn_step()

    got2 = one_jepa_coupled_learn_step()
    np.testing.assert_array_equal(got["q_eval_params"], got2["q_eval_params"])
    np.testing.assert_array_equal(got["encoder_params"], got2["encoder_params"])
    assert got["ddqn_loss"] == got2["ddqn_loss"]

    _assert_matches(
        got, ref,
        scalar_keys=("ddqn_loss", "jepa_pred_mse", "total_loss"),
        param_keys=("q_eval_params", "encoder_params"),
    )


def test_golden_jepa_coupled_unaffected_by_flag_on():
    """learn_DDQN_jepa_coupled has NO bypass (already _q_target-routed), so
    the flag — even ON — must not change it at neutral stabilizer values."""
    ref = np.load(os.path.join(_REFS_DIR, "e2e_arith_jepa_coupled.npz"))
    got = one_jepa_coupled_learn_step({"GSP_E2E_UNIFIED_TARGET_ARITH": True})
    _assert_matches(
        got, ref,
        scalar_keys=("ddqn_loss", "jepa_pred_mse", "total_loss"),
        param_keys=("q_eval_params", "encoder_params"),
    )
