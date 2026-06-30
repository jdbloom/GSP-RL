"""Freeze baseline DQN/DDQN learn-step references for the T4 golden gate.

Run from the repo root with:

    python tests/test_learning_aids/freeze_t4.py

T4 target: hoist per-call getattr(self, 'gsp_*', ...) lookups in learning_aids.py
and actor.py to __init__-time attrs. The numeric outputs must be bit-identical.
This freeze captures the PRE-OPTIMIZATION baseline so that after the patch the
test_golden_t4.py gate compares optimized code to the frozen baseline numbers.
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
    from golden_helpers import one_learn_step  # script-style
except ImportError:
    from tests.test_learning_aids.golden_helpers import one_learn_step


def main() -> None:
    refs_dir = os.path.join(_HERE, "golden_refs")
    os.makedirs(refs_dir, exist_ok=True)
    for net, fname in (("DQN", "t4_dqn.npz"), ("DDQN", "t4_ddqn.npz")):
        result = one_learn_step(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")


if __name__ == "__main__":
    main()
