"""Unit tests for the RunningStandardizer (opt-in GSP feature standardization).

These tests use synthetic/injected numbers only (no experiment data) so they are
zero-spend and deterministic.
"""

import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.feature_stats import RunningStandardizer


def test_standardized_batch_has_unit_variance():
    """(a) After the running stats have seen a sample distribution, standardizing
    a fresh draw from that same distribution yields ~zero-mean, ~unit-variance."""
    rng = np.random.default_rng(0)
    # Tiny-magnitude scalar feature, mirroring the observed GSP slot std ~0.024.
    train = rng.normal(loc=0.31, scale=0.024, size=(4096, 1))
    std = RunningStandardizer(dim=1)
    std.update(train)

    fresh = rng.normal(loc=0.31, scale=0.024, size=(4096, 1))
    out = std.standardize(fresh)
    # The fresh batch mean differs from the training mean by finite-sample noise
    # (~1/sqrt(N) in std-units); the point is it is O(0) not O(1/0.024)~40.
    assert abs(out.mean()) < 0.1
    assert out.std() == pytest.approx(1.0, abs=5e-2)


def test_welford_matches_offline_mean_var():
    """Batched Welford aggregate equals the offline mean/var of all samples,
    independent of how the samples were chunked."""
    rng = np.random.default_rng(1)
    data = rng.normal(loc=-2.0, scale=3.0, size=(1000, 3))
    std = RunningStandardizer(dim=3)
    for chunk in np.array_split(data, 7):
        std.update(chunk)
    np.testing.assert_allclose(std.mean, data.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(std.var, data.var(axis=0), rtol=1e-10)


def test_identity_before_first_update():
    """Before any update, standardize returns the input unchanged (identity)."""
    std = RunningStandardizer(dim=1)
    x_np = np.array([0.5], dtype=np.float32)
    assert std.standardize(x_np) is x_np  # same object, no allocation
    x_t = T.tensor([0.5])
    assert std.standardize(x_t) is x_t


def test_standardize_does_not_update_stats():
    """(d) standardize is read-only: repeated calls never change the stats."""
    std = RunningStandardizer(dim=1)
    std.update(np.array([[1.0], [3.0]]))
    m0, v0, c0 = std.mean.copy(), std.var.copy(), std.count
    for _ in range(5):
        std.standardize(np.array([[10.0], [20.0]]))
        std.standardize(T.tensor([[10.0], [20.0]]))
    np.testing.assert_array_equal(std.mean, m0)
    np.testing.assert_array_equal(std.var, v0)
    assert std.count == c0


def test_torch_and_numpy_paths_agree():
    """The torch path and numpy path produce the same standardized values."""
    std = RunningStandardizer(dim=2)
    std.update(np.array([[0.1, 5.0], [0.3, 7.0], [0.2, 6.0]]))
    x = np.array([[0.25, 6.5]], dtype=np.float32)
    np_out = std.standardize(x)
    t_out = std.standardize(T.tensor(x)).numpy()
    np.testing.assert_allclose(np_out, t_out, rtol=1e-5, atol=1e-6)


def test_torch_standardize_preserves_grad():
    """The learn-side path must keep the autograd edge (mean/std are constants)."""
    std = RunningStandardizer(dim=1)
    std.update(np.array([[0.1], [0.3], [0.5]]))
    x = T.tensor([[0.4]], requires_grad=True)
    out = std.standardize(x)
    assert out.requires_grad
    out.sum().backward()
    # d/dx of (x-mean)/s is 1/s > 0 — gradient flows and is finite.
    assert x.grad is not None
    assert T.isfinite(x.grad).all()
    assert float(x.grad.item()) > 0.0


def test_frozen_mean_ablation_composes_to_near_zero():
    """(ablation composition) The frozen_mean ablation replaces the live pred with
    the per-episode running MEAN of predictions. Standardizing that mean with
    stats whose mean ~matches it yields ~0 — i.e. the ablation still severs the
    signal (a constant near the feature's own mean) after normalization."""
    rng = np.random.default_rng(2)
    preds = rng.normal(loc=0.31, scale=0.024, size=(2048, 1))
    std = RunningStandardizer(dim=1)
    std.update(preds)
    # frozen_mean feeds a constant equal to the mean of predictions.
    frozen = np.array([[preds.mean()]], dtype=np.float32)
    out = std.standardize(frozen)
    assert abs(float(out[0, 0])) < 0.1


def test_scalar_1d_shape_handling():
    """dim==1 accepts a flat (N,) batch and treats it as (N, 1)."""
    std = RunningStandardizer(dim=1)
    std.update(np.array([1.0, 2.0, 3.0]))
    assert std.count == 3
    assert std.mean[0] == pytest.approx(2.0)


def test_update_rejects_wrong_width():
    std = RunningStandardizer(dim=3)
    with pytest.raises(ValueError):
        std.update(np.zeros((4, 2)))


def test_empty_batch_is_noop():
    std = RunningStandardizer(dim=1)
    std.update(np.zeros((0, 1)))
    assert std.count == 0
