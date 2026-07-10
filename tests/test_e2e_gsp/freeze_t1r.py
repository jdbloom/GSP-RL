"""Freeze the T1R golden references from the PRE-PATCH baseline.

Run ONCE on the baseline (before the T1R optimization patch is applied), then
commit the resulting ``golden_refs/*.npz`` together with the golden test:

    ~/.pyenv/versions/space/bin/python tests/test_e2e_gsp/freeze_t1r.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from golden_helpers_e2e import one_e2e_learn_step, one_jepa_coupled_learn_step  # noqa: E402

_REFS_DIR = os.path.join(_HERE, "golden_refs")


def main() -> None:
    os.makedirs(_REFS_DIR, exist_ok=True)
    e2e = one_e2e_learn_step()
    np.savez(os.path.join(_REFS_DIR, "t1r_e2e.npz"), **e2e)
    print(f"frozen t1r_e2e.npz: total_loss={e2e['total_loss']:.8f}")

    jepa = one_jepa_coupled_learn_step()
    np.savez(os.path.join(_REFS_DIR, "t1r_jepa.npz"), **jepa)
    print(f"frozen t1r_jepa.npz: total_loss={jepa['total_loss']:.8f}")


if __name__ == "__main__":
    main()
