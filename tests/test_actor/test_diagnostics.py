"""Tests for gsp_rl.src.actors.diagnostics — plasticity, rank, and Q-gap metrics.

Design: spec at docs/specs/2026-04-17-diagnostics-instrumentation.md.

Metrics under test:
- compute_fau: dormant-neuron fraction per layer (Sokar 2023 ReDo, τ_dead=0.1)
- compute_overactive_fau: over-active fraction per layer (Qin 2024 MARL-specific,
  τ_over=0.9 of max post-ReLU activation)
- compute_weight_norms: L2 norm of each layer's weight matrix
- compute_effective_rank: # singular values needed for 99% of activation variance
- compute_q_action_gap: Q(a*) - Q(a_next_best) over eval batch
- compute_gsp_pred_diversity: Shannon entropy of binned predictions

All functions are pure: inputs in, dict out, no side effects.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from gsp_rl.src.actors.diagnostics import (
    compute_fau,
    compute_overactive_fau,
    compute_weight_norms,
    compute_effective_rank,
    compute_q_action_gap,
    compute_gsp_pred_diversity,
)


# --------------------------------------------------------------------------------------
# Helper: build a simple MLP matching the production DDQN/DDPG shape
# --------------------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    """Same fc1/fc2/fc3 pattern as DDQN and DDPGActorNetwork (minus output activation)."""

    def __init__(self, input_size: int = 6, fc1: int = 16, fc2: int = 8, output: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, output)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)

    # Diagnostic-friendly accessor: penultimate post-ReLU activations
    def penultimate(self, x):
        h1 = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(h1))


# --------------------------------------------------------------------------------------
# FAU (dormant-neuron fraction)
# --------------------------------------------------------------------------------------

def test_fau_all_dead_returns_one():
    """If every post-ReLU activation is below tau_dead, FAU must be 1.0."""
    net = _TinyMLP()
    # Force all outputs <= 0: weight=0 and bias=-10 -> every output is -10 -> ReLU=0.
    # (Weight=-10 doesn't work because W*x_i is random for random x.)
    with torch.no_grad():
        for layer in (net.fc1, net.fc2):
            layer.weight.fill_(0.0)
            layer.bias.fill_(-10.0)
    batch = torch.randn(64, 6)
    result = compute_fau(net, batch, layer_names=["fc1", "fc2"], tau_dead=0.1)
    assert result["fau_fc1"] == 1.0
    assert result["fau_fc2"] == 1.0


def test_fau_all_active_returns_zero():
    """If every activation is > tau_dead, FAU must be 0.0."""
    net = _TinyMLP()
    with torch.no_grad():
        for layer in (net.fc1, net.fc2):
            layer.weight.fill_(0.0)
            layer.bias.fill_(1.0)  # All post-ReLU activations = 1.0 > 0.1
    batch = torch.randn(64, 6)
    result = compute_fau(net, batch, layer_names=["fc1", "fc2"], tau_dead=0.1)
    assert result["fau_fc1"] == 0.0
    assert result["fau_fc2"] == 0.0


def test_fau_half_dead_approximately_half():
    """Construct a layer where exactly half the units are dead."""
    net = _TinyMLP(input_size=4, fc1=10, fc2=4, output=2)
    with torch.no_grad():
        # First 5 units: positive bias → always active
        # Last 5 units: large-negative bias → always dead
        net.fc1.weight.fill_(0.0)
        net.fc1.bias[:5] = 1.0
        net.fc1.bias[5:] = -10.0
    batch = torch.randn(64, 4)
    result = compute_fau(net, batch, layer_names=["fc1"], tau_dead=0.1)
    assert abs(result["fau_fc1"] - 0.5) < 0.01


def test_fau_returns_correct_keys():
    """Output dict has exactly the requested layer names, prefixed with fau_."""
    net = _TinyMLP()
    batch = torch.randn(8, 6)
    result = compute_fau(net, batch, layer_names=["fc1", "fc2"])
    assert set(result.keys()) == {"fau_fc1", "fau_fc2"}


# --------------------------------------------------------------------------------------
# Over-active fraction (Qin 2024 MARL signal)
# --------------------------------------------------------------------------------------

def test_overactive_detects_saturated_units():
    """Units that always output near the max should count as over-active."""
    net = _TinyMLP(input_size=4, fc1=10, fc2=4, output=2)
    with torch.no_grad():
        # First 3 units saturate high; rest scattered
        net.fc1.weight.fill_(0.0)
        net.fc1.bias[:3] = 100.0  # Huge positive → always high
        net.fc1.bias[3:] = 0.1    # Modest
    batch = torch.randn(64, 4)
    result = compute_overactive_fau(net, batch, layer_names=["fc1"], tau_over=0.9)
    # 3 of 10 units saturated at max → overactive = 0.3
    assert abs(result["overactive_fc1"] - 0.3) < 0.05


def test_overactive_no_saturation_returns_low():
    """Balanced layer should have low over-active fraction."""
    net = _TinyMLP()
    with torch.no_grad():
        for layer in (net.fc1, net.fc2):
            layer.weight.normal_(0.0, 0.1)
            layer.bias.fill_(0.0)
    batch = torch.randn(64, 6)
    result = compute_overactive_fau(net, batch, layer_names=["fc1", "fc2"], tau_over=0.9)
    assert result["overactive_fc1"] <= 0.2
    assert result["overactive_fc2"] <= 0.2


# --------------------------------------------------------------------------------------
# Weight norms
# --------------------------------------------------------------------------------------

def test_weight_norms_identity_matrix():
    """Identity weight matrix: ‖W‖_F = √n."""
    net = _TinyMLP(input_size=4, fc1=4, fc2=4, output=2)
    with torch.no_grad():
        net.fc1.weight.copy_(torch.eye(4))
        net.fc1.bias.fill_(0.0)
    result = compute_weight_norms(net, layer_names=["fc1"])
    assert abs(result["wnorm_fc1"] - 2.0) < 1e-5  # √4 = 2.0


def test_weight_norms_zero_matrix():
    """Zero weight matrix: ‖W‖ = 0."""
    net = _TinyMLP()
    with torch.no_grad():
        net.fc1.weight.fill_(0.0)
    result = compute_weight_norms(net, layer_names=["fc1"])
    assert result["wnorm_fc1"] == 0.0


def test_weight_norms_returns_all_requested():
    """Dict has an entry for every requested layer."""
    net = _TinyMLP()
    result = compute_weight_norms(net, layer_names=["fc1", "fc2", "fc3"])
    assert set(result.keys()) == {"wnorm_fc1", "wnorm_fc2", "wnorm_fc3"}


# --------------------------------------------------------------------------------------
# Effective rank (99% threshold on singular values)
# --------------------------------------------------------------------------------------

def test_effective_rank_full_rank_batch():
    """Random full-rank activations: rank = dim(penult)."""
    net = _TinyMLP(input_size=4, fc1=10, fc2=8, output=2)
    with torch.no_grad():
        # Orthogonal-ish weights produce full-rank outputs
        net.fc1.weight.copy_(torch.randn(10, 4) * 0.5)
        net.fc1.bias.fill_(0.5)
        net.fc2.weight.copy_(torch.randn(8, 10) * 0.3)
        net.fc2.bias.fill_(0.2)
    batch = torch.randn(256, 4)
    erank = compute_effective_rank(net, batch, penultimate_fn="penultimate", threshold=0.99)
    # Should be close to 8 (the penultimate dim)
    assert 4 <= erank <= 8


def test_effective_rank_collapsed_to_one():
    """If the penultimate layer outputs identical vectors, rank should be ~1."""
    net = _TinyMLP(input_size=4, fc1=4, fc2=4, output=2)
    with torch.no_grad():
        # Make the whole trunk output a constant
        net.fc1.weight.fill_(0.0)
        net.fc1.bias.fill_(1.0)
        net.fc2.weight.fill_(0.0)
        net.fc2.bias.fill_(1.0)
    batch = torch.randn(64, 4)
    erank = compute_effective_rank(net, batch, penultimate_fn="penultimate", threshold=0.99)
    assert erank <= 1  # Constant outputs have rank 0 or 1 depending on impl


def test_effective_rank_low_rank_construction():
    """Constructed rank-3 activations return rank close to 3."""
    # Build a batch where penultimate outputs lie in a 3-d subspace
    n_samples, penult_dim = 64, 8
    # Generate data in rank-3 subspace
    basis = torch.randn(penult_dim, 3)
    coeffs = torch.randn(n_samples, 3)
    rank3_acts = coeffs @ basis.T + 0.5  # Small offset to keep post-ReLU positive

    class _StubNet(nn.Module):
        def __init__(self, acts):
            super().__init__()
            self._acts = acts
            # Dummy layers so compute_weight_norms etc don't break if called
            self.fc1 = nn.Linear(4, 8)

        def penultimate(self, x):
            # Ignore x, return fixed rank-3 acts
            return self._acts

    net = _StubNet(rank3_acts)
    batch = torch.randn(n_samples, 4)  # Shape doesn't matter — stub ignores it
    erank = compute_effective_rank(net, batch, penultimate_fn="penultimate", threshold=0.99)
    # Rank-3 subspace should give erank ∈ {3, 4} (threshold noise)
    assert 2 <= erank <= 4


# --------------------------------------------------------------------------------------
# Q-value action gap (Weng-Lee 2026 cooperation-collapse signal)
# --------------------------------------------------------------------------------------

def test_q_action_gap_known_values():
    """Construct Q-values with a known gap, verify the function returns it."""

    class _FixedQ(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            # First action Q=5.0, second Q=2.0, third Q=1.0, ...
            q = torch.tensor([[5.0, 2.0, 1.0, 0.5]]).expand(batch_size, 4).clone()
            return q

    net = _FixedQ()
    batch = torch.randn(32, 6)
    result = compute_q_action_gap(net, batch)
    # Best action: 5.0, next best: 2.0 → gap = 3.0
    assert abs(result["q_action_gap_mean"] - 3.0) < 1e-4
    assert abs(result["q_action_gap_std"] - 0.0) < 1e-4  # Same gap every sample
    assert abs(result["q_max_mean"] - 5.0) < 1e-4


def test_q_action_gap_collapsed_to_zero():
    """If all Q-values equal, gap should be 0."""

    class _FlatQ(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 4) * 3.0

    net = _FlatQ()
    batch = torch.randn(32, 6)
    result = compute_q_action_gap(net, batch)
    assert abs(result["q_action_gap_mean"]) < 1e-5


# --------------------------------------------------------------------------------------
# GSP prediction diversity (entropy)
# --------------------------------------------------------------------------------------

def test_pred_diversity_constant_is_zero():
    """All predictions same → entropy = 0."""
    preds = np.full(1000, 0.5, dtype=np.float32)
    entropy = compute_gsp_pred_diversity(preds, n_bins=10, low=-1.0, high=1.0)
    assert entropy < 0.01


def test_pred_diversity_uniform_is_log_n():
    """Uniformly distributed predictions → entropy ≈ log(n_bins)."""
    rng = np.random.default_rng(42)
    preds = rng.uniform(-1.0, 1.0, size=100_000).astype(np.float32)
    entropy = compute_gsp_pred_diversity(preds, n_bins=10, low=-1.0, high=1.0)
    # log(10) ≈ 2.3026; with sampling noise, expect > 2.25
    assert 2.25 < entropy < 2.31


def test_pred_diversity_returns_float():
    """Function returns Python float (for h5 compatibility)."""
    preds = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    entropy = compute_gsp_pred_diversity(preds)
    assert isinstance(entropy, float)
