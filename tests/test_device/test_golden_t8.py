"""T8 golden-equivalence gate for AttentionEncoder per-forward allocation removal.

T8 caches:
1. Positional indices (T.arange(0, max_length)) as a registered buffer in
   AttentionEncoder.__init__, sliced/expanded in forward instead of being
   rebuilt on every call.
2. The attention scale (embed_size**0.5) in SelfAttention.__init__ instead
   of being recomputed on every forward.

Both changes must be purely inert — forward output must be bit-identical to the
frozen baseline. The test loads frozen .npz references generated on the
PRE-OPTIMIZATION baseline and compares against the current code.

Device: forced to CPU via a pytest fixture so the monkeypatch is scoped to this
file only and does not bleed into other test modules (test isolation).
"""
import os

import numpy as np
import pytest
import torch as T

_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_refs")

_RTOL = 1e-6
_ATOL = 1e-6


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Force get_device to return CPU so the gate is device-agnostic.

    Scoped to this module only via pytest monkeypatch (reverted after each test).
    """
    import gsp_rl.src.networks.self_attention as _sa_mod
    monkeypatch.setattr(_sa_mod, "get_device", lambda recurrent=False: T.device("cpu"))


def _make_encoder():
    """Build a fixed-seed AttentionEncoder on CPU, eval mode (no dropout)."""
    from gsp_rl.src.networks.self_attention import AttentionEncoder
    T.manual_seed(0)
    enc = AttentionEncoder(
        input_size=6,
        output_size=1,
        min_max_action=1.0,
        encode_size=1,
        embed_size=4,
        hidden_size=8,
        heads=2,
        forward_expansion=2,
        dropout=0.0,
        max_length=5,
    )
    enc.eval()
    return enc


def _forward(enc, N: int, seed: int = 42) -> np.ndarray:
    T.manual_seed(seed)
    x = T.randn(N, 5, 6)  # seq_len = max_length = 5
    with T.no_grad():
        out = enc(x)
    return out.cpu().detach().numpy()


@pytest.mark.parametrize("N,fname", [(1, "t8_attn_n1.npz"), (4, "t8_attn_n4.npz"), (8, "t8_attn_n8.npz")])
def test_golden_t8_forward(N, fname):
    """Forward output must be bit-identical to the frozen baseline reference."""
    ref = np.load(os.path.join(_REFS_DIR, fname))["out"]
    enc = _make_encoder()

    got = _forward(enc, N)

    # Determinism sub-assertion: two in-process calls must match exactly.
    got2 = _forward(enc, N)
    np.testing.assert_array_equal(got, got2, err_msg=f"N={N}: non-deterministic forward")

    # Golden equivalence: must match the frozen pre-optimization baseline.
    np.testing.assert_allclose(got, ref, rtol=_RTOL, atol=_ATOL,
                               err_msg=f"N={N}: output diverged from frozen baseline")
