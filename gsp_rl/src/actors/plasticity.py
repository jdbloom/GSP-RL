"""ReDo — Recycling Dormant neurons (Sokar et al. 2023, arXiv 2302.12902).

Periodically recycles dead units to restore plasticity. A unit is dormant when
its post-ReLU activation is <= tau across a probe batch (the same criterion as
diagnostics.compute_fau). For each dormant unit d in layer L:
  - re-initialize its INCOMING weights (row d of W_L) + bias[d]  -> it can learn again
  - zero its OUTGOING weights (column d of W_{L+1})              -> no downstream disruption
  - clear the Adam optimizer state for the reset parameters      -> fresh moments

Inert by default: only invoked when REDO_ENABLED is set. The reset runs under
no_grad and clears its own probe gradients, so it does not perturb the training
step. Targets the actor/Q-network trunk (fc1, fc2) where dormancy was observed to
drive the gate-training seed variance (dormancy<->success r=-0.54).
"""
from __future__ import annotations

from typing import List, Tuple

import torch as T
import torch.nn as nn


def _dormant_mask(net: nn.Module, layer: str, batch: T.Tensor, tau: float) -> T.Tensor:
    """Boolean mask (per output unit of `layer`) of units whose post-ReLU
    activation never exceeds tau across the batch."""
    captured = {}

    def hook(_m, _inp, out):
        captured["a"] = T.relu(out).detach()

    h = getattr(net, layer).register_forward_hook(hook)
    try:
        with T.no_grad():
            net(batch.to(next(net.parameters()).device))
    finally:
        h.remove()
    acts = captured["a"]                       # (N, units)
    peak = acts.amax(dim=0)                     # max activation per unit over batch
    return peak <= tau


def _reset_adam(optimizer, param, rows=None, cols=None) -> None:
    """Zero the Adam moment estimates for the given rows/cols of `param`."""
    st = optimizer.state.get(param)
    if not st:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            if rows is not None:
                st[key][rows] = 0.0
            if cols is not None:
                st[key][:, cols] = 0.0


def redo_reset(net: nn.Module, batch: T.Tensor,
               layer_pairs: List[Tuple[str, str]], tau: float = 0.1) -> int:
    """Recycle dormant units across the given (layer, next_layer) pairs. Returns
    the number of units reset. `net` must expose `.optimizer` (Adam)."""
    optimizer = getattr(net, "optimizer", None)
    n_reset = 0
    with T.no_grad():
        for layer, nxt in layer_pairs:
            dead = _dormant_mask(net, layer, batch, tau)
            idx = T.nonzero(dead, as_tuple=False).flatten().tolist()
            if not idx:
                continue
            lin = getattr(net, layer)
            nxt_lin = getattr(net, nxt)
            for d in idx:
                # incoming: re-initialize this unit's row + bias
                nn.init.kaiming_uniform_(lin.weight[d:d + 1], a=5 ** 0.5)
                if lin.bias is not None:
                    lin.bias[d] = 0.0
                # outgoing: zero this unit's column in the next layer
                nxt_lin.weight[:, d] = 0.0
            if optimizer is not None:
                _reset_adam(optimizer, lin.weight, rows=idx)
                if lin.bias is not None:
                    _reset_adam(optimizer, lin.bias, rows=idx)
                _reset_adam(optimizer, nxt_lin.weight, cols=idx)
            n_reset += len(idx)
    return n_reset
