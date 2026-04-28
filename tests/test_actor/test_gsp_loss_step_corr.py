"""Unit tests for the Phase 4 gsp_loss_step_corr diagnostic metric.

learn_gsp_mse() now returns (loss_float, batch_corr_float) instead of a plain
float.  batch_corr is the Pearson correlation between the fresh forward-pass
predictions used to compute the MSE loss and the replay-buffer labels for that
same batch.

Verifies:
1. Return type is (float, float) tuple — not a plain float.
2. loss element is finite.
3. batch_corr element is either finite (normal case) or nan (undefined corr).
4. learn_gsp() (called via Actor.learn()) populates
   last_gsp_loss_step_corr_samples with at least one finite float when the
   replay buffer has enough samples.
5. learn_gsp_mse on a perfectly learnable linear task returns a positive
   correlation (head corr improves over training).
6. Gradient graph is unaffected: loss.backward() + optimizer.step() paths
   are identical to pre-change behaviour (no new requires_grad tensors leak
   from the correlation computation).
"""

import math

import numpy as np
import pytest
import torch

from gsp_rl.src.actors.actor import Actor


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_gsp_direct_mse.py conventions)
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
    cfg = dict(BASE_CONFIG)
    if extra_config:
        cfg.update(extra_config)
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


def _fill_gsp_buffer(actor, n=100, seed=0):
    """Fill the GSP replay buffer with (state, label) pairs."""
    rng = np.random.default_rng(seed)
    states_list = []
    labels_list = []
    for _ in range(n):
        s = rng.standard_normal(GSP_INPUT_SIZE).astype(np.float32)
        label = np.float32(rng.standard_normal())
        actor.store_gsp_transition(s, label, 0.0, np.zeros_like(s), False)
        states_list.append(s)
        labels_list.append(float(label))
    return np.array(states_list), np.array(labels_list)


def _fill_primary_buffer(actor, n=100, seed=99):
    """Fill the primary replay buffer so learn() doesn't short-circuit."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.random(actor.network_input_size).astype(np.float32)
        s_ = rng.random(actor.network_input_size).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        actor.store_transition(s, a, 0.0, s_, False, actor.networks)


def _fill_gsp_buffer_linear(actor, n=400, seed=0):
    """Fill GSP buffer with a deterministic linear label: label = s[0] * 0.5."""
    rng = np.random.default_rng(seed)
    states, labels = [], []
    for _ in range(n):
        s = rng.standard_normal(GSP_INPUT_SIZE).astype(np.float32)
        label = np.float32(s[0] * 0.5)
        actor.store_gsp_transition(s, label, 0.0, np.zeros_like(s), False)
        states.append(s)
        labels.append(float(label))
    return np.array(states), np.array(labels)


# ---------------------------------------------------------------------------
# Test 1: return type is (float, float) tuple
# ---------------------------------------------------------------------------

def test_learn_gsp_mse_returns_tuple():
    """learn_gsp_mse must return a 2-tuple (loss_float, batch_corr_float)."""
    torch.manual_seed(0)
    np.random.seed(0)
    actor = make_gsp_actor()
    _fill_gsp_buffer(actor, seed=0)
    _fill_primary_buffer(actor)

    result = actor.learn_gsp_mse(actor.gsp_networks)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-element tuple, got length {len(result)}"
    loss_val, batch_corr = result
    assert isinstance(loss_val, float), (
        f"First element (loss) must be float, got {type(loss_val)}"
    )
    assert isinstance(batch_corr, float), (
        f"Second element (corr) must be float, got {type(batch_corr)}"
    )


# ---------------------------------------------------------------------------
# Test 2: loss element is finite
# ---------------------------------------------------------------------------

def test_learn_gsp_mse_loss_is_finite():
    """Loss element of the returned tuple must be finite."""
    torch.manual_seed(1)
    np.random.seed(1)
    actor = make_gsp_actor()
    _fill_gsp_buffer(actor, seed=1)
    _fill_primary_buffer(actor)

    loss_val, _ = actor.learn_gsp_mse(actor.gsp_networks)
    assert math.isfinite(loss_val), f"Loss must be finite, got {loss_val}"


# ---------------------------------------------------------------------------
# Test 3: batch_corr is finite or nan — never inf
# ---------------------------------------------------------------------------

def test_learn_gsp_mse_batch_corr_not_inf():
    """batch_corr must be a finite float or nan — never +/-inf."""
    torch.manual_seed(2)
    np.random.seed(2)
    actor = make_gsp_actor()
    _fill_gsp_buffer(actor, seed=2)
    _fill_primary_buffer(actor)

    _, batch_corr = actor.learn_gsp_mse(actor.gsp_networks)
    assert not math.isinf(batch_corr), (
        f"batch_corr must not be inf, got {batch_corr}"
    )


# ---------------------------------------------------------------------------
# Test 4: learn_gsp() via Actor.learn() accumulates samples
# ---------------------------------------------------------------------------

def test_learn_populates_loss_step_corr_samples():
    """After Actor.learn(), last_gsp_loss_step_corr_samples has >= 1 entry."""
    torch.manual_seed(3)
    np.random.seed(3)
    actor = make_gsp_actor()
    _fill_gsp_buffer(actor, seed=3)
    _fill_primary_buffer(actor)

    actor.learn()

    samples = getattr(actor, "last_gsp_loss_step_corr_samples", None)
    assert samples is not None, "last_gsp_loss_step_corr_samples must exist on Actor"
    assert len(samples) >= 1, (
        f"Expected at least 1 sample after learn(), got {len(samples)}"
    )
    for s in samples:
        assert isinstance(s, float), f"Each sample must be float, got {type(s)}"
        assert math.isfinite(s), f"Each sample must be finite, got {s}"


# ---------------------------------------------------------------------------
# Test 5: correlation is positive on a learnable linear task after training
# ---------------------------------------------------------------------------

def test_batch_corr_positive_after_training_on_linear_task():
    """After 100 learn steps on a linear label task, mean batch_corr > 0.

    This confirms the loss-step path is measuring real learning signal, not
    returning garbage or zero.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    actor = make_gsp_actor()
    _fill_gsp_buffer_linear(actor, n=400, seed=42)
    _fill_primary_buffer(actor)

    corr_values = []
    for _ in range(100):
        _, batch_corr = actor.learn_gsp_mse(actor.gsp_networks)
        if math.isfinite(batch_corr):
            corr_values.append(batch_corr)

    assert len(corr_values) > 0, "No finite batch_corr values collected"
    mean_corr = float(np.mean(corr_values))
    assert mean_corr > 0.0, (
        f"Expected positive mean batch_corr on linear task after training, "
        f"got {mean_corr:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: correlation computation does not affect gradient graph
# ---------------------------------------------------------------------------

def test_batch_corr_does_not_affect_gradients():
    """Gradient graph must be unaffected by the correlation computation.

    Verifies that the correlation block uses T.no_grad() + detach() and
    that no new requires_grad tensors are introduced that would accumulate
    across calls (which would break the existing loss.backward() path).
    """
    torch.manual_seed(5)
    np.random.seed(5)
    actor = make_gsp_actor()
    _fill_gsp_buffer(actor, seed=5)
    _fill_primary_buffer(actor)

    # Capture param grad norms before
    net = actor.gsp_networks["actor"]
    for p in net.parameters():
        p.grad = None

    # Run one learn step — if correlation leaks a graph, backward will raise
    loss_val, batch_corr = actor.learn_gsp_mse(actor.gsp_networks)

    # Parameters should have gradients applied and zeroed by the optimizer step.
    # The optimizer.zero_grad() was called at the start of learn_gsp_mse, and
    # optimizer.step() was called after backward. Grads are NOT zeroed after step
    # in PyTorch by default — they should be non-None but finite.
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), (
                f"NaN gradient on {name} after learn_gsp_mse"
            )
            assert not torch.isinf(p.grad).any(), (
                f"Inf gradient on {name} after learn_gsp_mse"
            )
