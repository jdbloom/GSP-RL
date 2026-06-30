"""Freeze baseline DQN/DDQN/DDPG/TD3 learn-step references for the T3 golden gate.

Run from the repo root with:

    python tests/test_learning_aids/freeze_t3.py

T3 target (partial — _check_nan combine only): replace the two-branch
isinstance+isnan+isinf check with a single (isnan|isinf).any() call.  The
change is purely inert: same logical result, one fewer GPU sync per check.
All learn paths that call _check_nan must remain bit-exact.

This freeze captures the PRE-OPTIMIZATION baseline so that after the patch
test_golden_t3.py compares optimized code to the frozen baseline numbers.
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

    for net, fname in (("DQN", "t3_dqn.npz"), ("DDQN", "t3_ddqn.npz")):
        result = one_learn_step(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")

    for net, fname in (("DDPG", "t3_ddpg.npz"), ("TD3", "t3_td3.npz")):
        result = one_learn_step_continuous(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")


if __name__ == "__main__":
    main()
