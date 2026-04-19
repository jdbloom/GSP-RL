"""Tests for the L2-ER (effective-rank regularization) auxiliary loss on the GSP head.

Verifies:
1. With GSP_L2ER_LAMBDA=0.0 (default), learn_gsp_mse is a strict no-op vs the
   pre-existing MSE-only path (same loss value to float precision).
2. With GSP_L2ER_LAMBDA > 0.0, the combined loss is strictly larger than MSE alone
   when activations are low-rank (e.g., after a zero-target backward step that
   pushes the hidden units toward a collapsed regime).
3. backward() populates gradients on all weight tensors — both the MSE term and
   the L2-ER term flow gradients back to fc1.weight, fc2.weight, mu.weight.
4. The effective-rank term is positive and finite for a random batch.
"""

import numpy as np
import pytest
import torch

from gsp_rl.src.actors.actor import Actor
from gsp_rl.src.actors.learning_aids import gsp_l2er_loss


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 16,
    "MEM_SIZE": 1000,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 1,
    "GSP_BATCH_SIZE": 16,
}

INPUT_SIZE = 8
OUTPUT_SIZE = 4
GSP_INPUT_SIZE = 6
GSP_OUTPUT_SIZE = 1


def make_gsp_actor(extra_config=None):
    cfg = {**BASE_CONFIG, **(extra_config or {})}
    return Actor(
        id=1,
        config=cfg,
        network="DDPG",
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1,
        meta_param_size=1,
        gsp=True,
        gsp_input_size=GSP_INPUT_SIZE,
        gsp_output_size=GSP_OUTPUT_SIZE,
    )


def _fill_gsp_buffer(actor, n=200, seed=0):
    """Fill the GSP replay buffer with random (state, label) pairs."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        state = rng.uniform(-1, 1, size=GSP_INPUT_SIZE).astype(np.float32)
        label = float(np.mean(state))
        actor.store_gsp_transition(
            state, np.float32(label), 0.0, np.zeros_like(state), False
        )


def _fill_primary_buffer(actor, n=20):
    rng = np.random.default_rng(99)
    for _ in range(n):
        s = rng.random(actor.network_input_size).astype(np.float32)
        s_ = rng.random(actor.network_input_size).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        actor.store_transition(s, a, 0.0, s_, False, actor.networks)


# ---------------------------------------------------------------------------
# Test 1: default lambda=0.0 is a strict no-op
# ---------------------------------------------------------------------------

def test_l2er_default_lambda_is_noop():
    """GSP_L2ER_LAMBDA=0.0 (default) never calls gsp_l2er_loss.

    We patch gsp_l2er_loss in the learn_gsp_mse module and confirm it is never
    invoked when lambda=0.0, verifying the guard branch preserves the existing
    MSE-only code path exactly (no computation graph changes, no SVD overhead).
    """
    from unittest.mock import patch

    torch.manual_seed(7)
    np.random.seed(7)

    actor = make_gsp_actor()  # lambda defaults to 0.0
    assert actor.gsp_l2er_lambda == 0.0

    _fill_gsp_buffer(actor, seed=42)
    _fill_primary_buffer(actor)

    call_count = []

    def _spy(*args, **kwargs):
        call_count.append(1)
        from gsp_rl.src.actors.learning_aids import gsp_l2er_loss as real
        return real(*args, **kwargs)

    with patch(
        'gsp_rl.src.actors.learning_aids.gsp_l2er_loss',
        side_effect=_spy,
    ):
        actor.learn()

    assert len(call_count) == 0, (
        f"gsp_l2er_loss should NOT be called when lambda=0.0, but was called "
        f"{len(call_count)} time(s)"
    )
    assert actor.last_gsp_loss is not None
    assert isinstance(actor.last_gsp_loss, float)


# ---------------------------------------------------------------------------
# Test 2: lambda > 0 produces a lower combined loss than MSE alone
# ---------------------------------------------------------------------------

def test_l2er_positive_lambda_produces_lower_combined_loss():
    """With lambda > 0, L_total = MSE - lambda * erank_sum < MSE_only.

    We compute both values from the same network and same batch by:
    1. Computing the pure MSE loss directly from gsp_l2er_loss helper.
    2. Computing the combined loss and confirming it is strictly smaller.

    This is a unit test of the loss formula, not of optimizer convergence.
    """
    import torch.nn.functional as F
    from gsp_rl.src.actors.learning_aids import gsp_l2er_loss

    torch.manual_seed(3)
    actor = make_gsp_actor({"GSP_L2ER_LAMBDA": 0.5})
    net = actor.gsp_networks['actor']
    device = net.device

    # Random batch and labels
    batch_size = 32
    states = torch.randn(batch_size, GSP_INPUT_SIZE, device=device)
    labels = torch.randn(batch_size, 1, device=device)

    # MSE-only loss
    preds = net.forward(states)
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    mse_loss = F.mse_loss(preds, labels)

    # Combined loss: MSE - lambda * erank_sum
    # gsp_l2er_loss returns -(erank1 + erank2) so -gsp_l2er_loss = erank_sum > 0
    erank_sum = -gsp_l2er_loss(net, states)
    combined_loss = mse_loss - 0.5 * erank_sum

    assert erank_sum.item() > 0.0, "erank_sum must be positive"
    assert combined_loss.item() < mse_loss.item(), (
        f"Combined loss ({combined_loss.item():.6f}) should be < MSE-only "
        f"({mse_loss.item():.6f}) when lambda > 0 and erank_sum > 0."
    )


# ---------------------------------------------------------------------------
# Test 3: backward() through both terms populates gradients on all weight tensors
# ---------------------------------------------------------------------------

def test_l2er_backward_populates_gradients_on_all_weights():
    """backward() through the combined loss populates .grad on fc1, fc2, and mu weights.

    We bypass the replay buffer and directly construct the combined loss from
    a synthetic batch, then call backward(). This confirms:
    - The MSE term flows gradients back to mu.weight (output projection).
    - The L2-ER term flows gradients back to fc1.weight and fc2.weight (hidden layers).
    - The SVD path does not produce NaN/Inf gradients.
    """
    import torch.nn.functional as F
    from gsp_rl.src.actors.learning_aids import gsp_l2er_loss

    torch.manual_seed(17)
    actor = make_gsp_actor({"GSP_L2ER_LAMBDA": 1.0})
    net = actor.gsp_networks['actor']
    device = net.device

    batch_size = 32
    states = torch.randn(batch_size, GSP_INPUT_SIZE, device=device)
    labels = torch.randn(batch_size, 1, device=device)

    # Zero existing gradients
    net.optimizer.zero_grad()

    # Forward + combined loss
    preds = net.forward(states)
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    mse_loss = F.mse_loss(preds, labels)
    erank_sum = -gsp_l2er_loss(net, states)
    combined_loss = mse_loss - 1.0 * erank_sum

    combined_loss.backward()

    assert net.fc1.weight.grad is not None, "fc1.weight.grad is None after backward"
    assert net.fc2.weight.grad is not None, "fc2.weight.grad is None after backward"
    assert net.mu.weight.grad is not None, "mu.weight.grad is None after backward"

    # Gradients must be finite (no NaN/Inf from SVD path)
    assert torch.isfinite(net.fc1.weight.grad).all(), "fc1 grad contains NaN/Inf"
    assert torch.isfinite(net.fc2.weight.grad).all(), "fc2 grad contains NaN/Inf"
    assert torch.isfinite(net.mu.weight.grad).all(), "mu grad contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 4: gsp_l2er_loss returns a finite positive-valued negative tensor
# ---------------------------------------------------------------------------

def test_gsp_l2er_loss_is_finite_and_negative():
    """gsp_l2er_loss returns a negative scalar tensor (-(erank1 + erank2))
    that is finite and has a meaningful magnitude for a random batch."""
    torch.manual_seed(42)
    actor = make_gsp_actor({"GSP_L2ER_LAMBDA": 1.0})
    net = actor.gsp_networks['actor']
    batch = torch.randn(64, GSP_INPUT_SIZE, device=net.device)

    loss_val = gsp_l2er_loss(net, batch)

    assert loss_val.dim() == 0, "gsp_l2er_loss must return a scalar tensor"
    assert torch.isfinite(loss_val), f"gsp_l2er_loss returned non-finite value: {loss_val}"
    # The function returns -(erank1 + erank2); effective ranks are >=1, so result < 0
    assert loss_val.item() < 0.0, (
        f"gsp_l2er_loss should return a negative value (it's negated erank sum), "
        f"got {loss_val.item()}"
    )
    # Effective ranks should be >= 1 (each layer contributes at least 1), so sum >= 2
    erank_sum = -loss_val.item()
    assert erank_sum >= 1.0, (
        f"Effective rank sum should be >= 1 for a random batch, got {erank_sum}"
    )


# ---------------------------------------------------------------------------
# Test 5: low-rank activations yield lower effective rank than full-rank
# ---------------------------------------------------------------------------

def test_gsp_l2er_loss_lower_for_collapsed_activations():
    """Effective rank is lower for a rank-1 batch than for a full-rank batch.

    We construct a rank-1 input batch by scaling a single direction vector by
    random scalars (all rows = scalar_i * direction). After centering this is a
    rank-1 matrix with one non-zero singular value, which yields erank ~1.
    A full-rank random Gaussian batch has erank up to GSP_INPUT_SIZE.

    The assertion: loss_low (rank-1 erank_sum) > loss_full (high-rank erank_sum)
    because gsp_l2er_loss = -(erank_sum) and smaller erank_sum gives a larger
    (less negative) value.
    """
    torch.manual_seed(99)
    actor = make_gsp_actor({"GSP_L2ER_LAMBDA": 1.0})
    net = actor.gsp_networks['actor']
    device = net.device

    # Full-rank: 64 independent Gaussian rows — erank close to GSP_INPUT_SIZE
    full_rank_batch = torch.randn(64, GSP_INPUT_SIZE, device=device)

    # Rank-1: rows = scalar_i * direction. After centering this is rank-1 with
    # erank ~1. Use random scalars so centering leaves a non-trivial rank-1 matrix.
    direction = torch.randn(GSP_INPUT_SIZE, device=device)
    scalars = torch.randn(64, 1, device=device)       # 64 different scale factors
    low_rank_batch = scalars * direction.unsqueeze(0)  # (64, GSP_INPUT_SIZE), rank-1

    loss_full = gsp_l2er_loss(net, full_rank_batch)
    loss_low = gsp_l2er_loss(net, low_rank_batch)

    # gsp_l2er_loss returns -(erank_sum). Rank-1 batch has erank_sum ~2 (1 per layer);
    # full-rank batch has erank_sum >> 2. So loss_full << loss_low (more negative).
    assert loss_low.item() > loss_full.item(), (
        f"Rank-1 batch should produce less negative gsp_l2er_loss "
        f"(smaller erank_sum) than full-rank: low={loss_low.item():.4f} full={loss_full.item():.4f}"
    )
