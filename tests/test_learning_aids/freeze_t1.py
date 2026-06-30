"""Freeze baseline DQN/DDQN learn-step references for the T1 golden gate.

Run from the repo root with either:

    python tests/test_learning_aids/freeze_t1.py
    python -m tests.test_learning_aids.freeze_t1   # only if tests is a package

The script computes one deterministic learn step for DQN and DDQN on the CURRENT
(baseline) code and writes ``loss`` + post-step ``params`` to
``golden_refs/t1_dqn.npz`` and ``golden_refs/t1_ddqn.npz``.
"""
import os
import sys

import numpy as np

# Allow ``python tests/test_learning_aids/freeze_t1.py`` (no package) by putting
# this directory on sys.path so ``golden_helpers`` resolves either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Repo root (two levels up) so ``gsp_rl`` resolves under a bare ``python`` run.
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from golden_helpers import one_learn_step  # script-style invocation
except ImportError:  # pragma: no cover - package-style invocation
    from tests.test_learning_aids.golden_helpers import one_learn_step


def main() -> None:
    refs_dir = os.path.join(_HERE, "golden_refs")
    os.makedirs(refs_dir, exist_ok=True)
    for net, fname in (("DQN", "t1_dqn.npz"), ("DDQN", "t1_ddqn.npz")):
        result = one_learn_step(net)
        path = os.path.join(refs_dir, fname)
        np.savez(path, loss=result["loss"], params=result["params"])
        print(f"{net}: loss={result['loss']!r} params_shape={result['params'].shape} -> {path}")


if __name__ == "__main__":
    main()
