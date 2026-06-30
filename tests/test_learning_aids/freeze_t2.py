"""Freeze baseline learn-step references for the T2 golden gate.

Run from the repo root with:

    python tests/test_learning_aids/freeze_t2.py

T2 target: replace T.tensor(numpy).to(device) double-copies with
T.as_tensor(...).to(device) in sample_memory and the per-scheme sample blocks,
eliminating the intermediate CPU allocation for float32 arrays, and cache the
zero gsp_obs vector in ReplayBuffer.store_transition.  All changes are INERT
(bit-exact).

This freeze captures the PRE-OPTIMIZATION baseline for DQN, DDQN, DDPG, and
TD3 learn steps so that after the patch test_golden_t2.py compares optimized
code to the frozen baseline numbers.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from golden_helpers import one_learn_step, one_learn_step_continuous
except ImportError:
    from tests.test_learning_aids.golden_helpers import (
        one_learn_step,
        one_learn_step_continuous,
    )


def main() -> None:
    refs_dir = os.path.join(_HERE, "golden_refs")
    os.makedirs(refs_dir, exist_ok=True)

    for net, fname in (("DQN", "t2_dqn.npz"), ("DDQN", "t2_ddqn.npz")):
        result = one_learn_step(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")

    for net, fname in (("DDPG", "t2_ddpg.npz"), ("TD3", "t2_td3.npz")):
        result = one_learn_step_continuous(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")


if __name__ == "__main__":
    main()
