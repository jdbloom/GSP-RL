"""Freeze baseline AttentionEncoder forward-pass references for the T8 golden gate.

Run from the repo root with:

    python tests/test_device/freeze_t8.py

T8 target: cache T.arange positional indices as a registered buffer in
AttentionEncoder.__init__ (instead of rebuilding every forward), and cache
scale = embed_size**0.5 in SelfAttention.__init__. Forward output must be
bit-identical.

This freeze captures the PRE-OPTIMIZATION baseline on CPU so that after the
patch, test_golden_t8.py compares optimized code to the frozen baseline numbers.

Device: forced to CPU via monkeypatch so the reference is portable (no MPS/CUDA).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch as T

# Force CPU so the frozen reference is device-agnostic.
import gsp_rl.src.networks.self_attention as _sa_mod
_sa_mod.get_device = lambda recurrent=False: T.device("cpu")

from gsp_rl.src.networks.self_attention import AttentionEncoder


def make_encoder() -> AttentionEncoder:
    """Build a fixed-seed AttentionEncoder on CPU with eval mode (no dropout)."""
    T.manual_seed(0)
    enc = AttentionEncoder(
        input_size=6,
        output_size=1,
        min_max_action=1.0,
        encode_size=1,        # unused, kept for interface compat
        embed_size=4,
        hidden_size=8,
        heads=2,
        forward_expansion=2,
        dropout=0.0,          # deterministic, no stochastic path
        max_length=5,
    )
    enc.eval()
    return enc


def forward_cpu(enc: AttentionEncoder, N: int, seed: int = 42) -> np.ndarray:
    """Run one forward pass, seeded, return numpy output."""
    T.manual_seed(seed)
    x = T.randn(N, 5, 6)  # seq_len fixed at max_length=5
    with T.no_grad():
        out = enc(x)
    return out.cpu().detach().numpy()


def main() -> None:
    refs_dir = os.path.join(_HERE, "golden_refs")
    os.makedirs(refs_dir, exist_ok=True)

    enc = make_encoder()

    for N, fname in ((1, "t8_attn_n1.npz"), (4, "t8_attn_n4.npz"), (8, "t8_attn_n8.npz")):
        out = forward_cpu(enc, N)
        path = os.path.join(refs_dir, fname)
        np.savez(path, out=out)
        print(f"N={N}: shape={out.shape} first={out.ravel()[0]!r} -> {path}")


if __name__ == "__main__":
    main()
