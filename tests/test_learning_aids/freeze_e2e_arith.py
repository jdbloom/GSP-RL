"""Freeze the PRE-GSP_E2E_UNIFIED_TARGET_ARITH behavior of learn_DDQN_e2e and
learn_DDQN_jepa_coupled into golden_refs/, gating the flag-off path.

Run ONCE from the pre-change tree (this was executed at the merge-base of
fix/e2e-target-arith-parity, i.e. main BEFORE the flag existed):

    cd <repo root>
    PYTHONPATH=tests/test_learning_aids python tests/test_learning_aids/freeze_e2e_arith.py

test_golden_e2e_arith.py then asserts the current default-flag code reproduces
these references exactly-to-tolerance.
"""
import os

import numpy as np

from e2e_arith_helpers import one_e2e_learn_step, one_jepa_coupled_learn_step

_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_refs")


def main() -> None:
    os.makedirs(_REFS_DIR, exist_ok=True)

    e2e = one_e2e_learn_step()
    np.savez(
        os.path.join(_REFS_DIR, "e2e_arith_ddqn_e2e.npz"),
        ddqn_loss=e2e["ddqn_loss"],
        gsp_mse_loss=e2e["gsp_mse_loss"],
        total_loss=e2e["total_loss"],
        q_eval_params=e2e["q_eval_params"],
        gsp_actor_params=e2e["gsp_actor_params"],
    )
    print("froze e2e_arith_ddqn_e2e.npz",
          {k: e2e[k] for k in ("ddqn_loss", "gsp_mse_loss", "total_loss")})

    jc = one_jepa_coupled_learn_step()
    np.savez(
        os.path.join(_REFS_DIR, "e2e_arith_jepa_coupled.npz"),
        ddqn_loss=jc["ddqn_loss"],
        jepa_pred_mse=jc["jepa_pred_mse"],
        total_loss=jc["total_loss"],
        q_eval_params=jc["q_eval_params"],
        encoder_params=jc["encoder_params"],
    )
    print("froze e2e_arith_jepa_coupled.npz",
          {k: jc[k] for k in ("ddqn_loss", "jepa_pred_mse", "total_loss")})


if __name__ == "__main__":
    main()
