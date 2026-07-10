"""T1R golden-equivalence gate — e2e/jepa-coupled index-tensor optimization.

T1 residual: ``learn_DDQN_e2e`` and ``learn_DDQN_jepa_coupled`` still build
``T.LongTensor(np.arange(self.batch_size).astype(np.int64))`` per call and cast
actions via ``actions.type(T.LongTensor)`` (the sites T1 #11 converted in
learn_DQN/learn_DDQN/learn_DDQN_sf but not on the e2e paths). The replacement
(``T.arange(self.batch_size, device=...)`` + ``actions.long()``) is INERT:
identical index values, identical gather, identical gradients.

Each test loads a frozen baseline reference (losses + post-step params)
captured on the PRE-PATCH baseline by ``freeze_t1r.py`` and committed, then
recomputes the same deterministic learn step on the current code. Because the
optimization touches only index-tensor construction (no fp math reordered),
params must match the frozen baseline BIT-EXACTLY on CPU.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REFS_DIR = os.path.join(_HERE, "golden_refs")


def _assert_bitexact(got: dict, ref) -> None:
    for key in ref.files:
        np.testing.assert_array_equal(
            np.asarray(got[key]), ref[key],
            err_msg=f"'{key}' diverged from frozen baseline (change is NOT inert)",
        )


def test_golden_t1r_e2e_learn_step():
    from golden_helpers_e2e import one_e2e_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t1r_e2e.npz"))

    got = one_e2e_learn_step()

    # Determinism self-check: two fresh runs must agree bit-exactly.
    got2 = one_e2e_learn_step()
    np.testing.assert_array_equal(got["q_eval_params"], got2["q_eval_params"])
    assert got["total_loss"] == got2["total_loss"], \
        "learn_DDQN_e2e step is not deterministic"

    _assert_bitexact(got, ref)


def test_golden_t1r_jepa_coupled_learn_step():
    from golden_helpers_e2e import one_jepa_coupled_learn_step
    ref = np.load(os.path.join(_REFS_DIR, "t1r_jepa.npz"))

    got = one_jepa_coupled_learn_step()

    got2 = one_jepa_coupled_learn_step()
    np.testing.assert_array_equal(got["q_eval_params"], got2["q_eval_params"])
    np.testing.assert_array_equal(got["encoder_params"], got2["encoder_params"])
    assert got["total_loss"] == got2["total_loss"], \
        "learn_DDQN_jepa_coupled step is not deterministic"

    _assert_bitexact(got, ref)
