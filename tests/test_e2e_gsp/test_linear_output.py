"""Tests for DDPGActorNetwork use_linear_output flag.

Verifies that use_linear_output=True replaces tanh with a clamp, producing
larger gradients on the output layer — important for end-to-end GSP training
where tanh saturates near ±1 and near 0 (DIAL/CommNet/TarMAC motivation).
"""
import copy
import torch as T
import pytest

from gsp_rl.src.networks.ddpg import DDPGActorNetwork


INPUT_SIZE = 8
OUTPUT_SIZE = 3
LR = 1e-3
MIN_MAX_ACTION = 1.0


def make_actor(use_linear_output: bool = False) -> DDPGActorNetwork:
    return DDPGActorNetwork(
        id=0,
        lr=LR,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=MIN_MAX_ACTION,
        use_linear_output=use_linear_output,
    )


class TestTanhOutputBounded:
    """Default (use_linear_output=False) bounds output via tanh * min_max_action."""

    def test_tanh_output_bounded(self):
        net = make_actor(use_linear_output=False)
        # Large input to push pre-tanh activations far from zero
        x = T.ones(16, INPUT_SIZE, device=net.device) * 100.0
        with T.no_grad():
            out = net(x)
        assert out.shape == (16, OUTPUT_SIZE)
        # tanh saturates: all values strictly within (-1, 1) when scaled by 1.0
        assert (out.abs() < MIN_MAX_ACTION).all(), (
            "tanh output must be strictly less than min_max_action in magnitude"
        )

    def test_tanh_output_bounded_negative(self):
        net = make_actor(use_linear_output=False)
        x = T.ones(16, INPUT_SIZE, device=net.device) * -100.0
        with T.no_grad():
            out = net(x)
        assert (out.abs() < MIN_MAX_ACTION).all()


class TestLinearOutputClamped:
    """use_linear_output=True bounds output via clamp(-1, 1)."""

    def test_linear_output_clamped_positive(self):
        net = make_actor(use_linear_output=True)
        x = T.ones(16, INPUT_SIZE, device=net.device) * 100.0
        with T.no_grad():
            out = net(x)
        assert out.shape == (16, OUTPUT_SIZE)
        assert (out <= MIN_MAX_ACTION).all(), "clamp upper bound violated"
        assert (out >= -MIN_MAX_ACTION).all(), "clamp lower bound violated"

    def test_linear_output_clamped_negative(self):
        net = make_actor(use_linear_output=True)
        x = T.ones(16, INPUT_SIZE, device=net.device) * -100.0
        with T.no_grad():
            out = net(x)
        assert (out <= MIN_MAX_ACTION).all()
        assert (out >= -MIN_MAX_ACTION).all()

    def test_linear_output_passes_through_small_values(self):
        """For small mu values (no saturation), clamp is identity so out == mu_raw."""
        net = make_actor(use_linear_output=True)
        # Zero input -> near-zero mu due to init_w=3e-3 (bias is also near zero
        # since PyTorch linear default init is uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)))
        x = T.zeros(1, INPUT_SIZE, device=net.device)
        with T.no_grad():
            # Replicate forward() manually to get mu before clamping
            h = net.relu(net.fc1(x))
            h = net.relu(net.fc2(h))
            mu_raw = net.mu(h)
            out = net(x)
        # Clamp is identity when mu is within [-min_max_action, min_max_action]
        assert (mu_raw.abs() <= MIN_MAX_ACTION).all(), (
            "Precondition: mu_raw should be within bounds for this test to be meaningful"
        )
        T.testing.assert_close(out, mu_raw, atol=1e-6, rtol=0.0)


class TestLinearOutputHasLargerGradient:
    """Linear output produces >= gradient magnitude on the output layer vs tanh.

    tanh'(x) = 1 - tanh(x)^2 <= 1, so tanh always attenuates gradients.
    A clamp in the non-saturating region has gradient 1, which is >= tanh's
    gradient everywhere. We verify this with identical weights and a
    loss that would saturate tanh (large activations).
    """

    def test_linear_output_has_larger_gradient(self):
        tanh_net = make_actor(use_linear_output=False)
        linear_net = make_actor(use_linear_output=True)

        # Copy identical weights so only the output activation differs
        linear_net.load_state_dict(copy.deepcopy(tanh_net.state_dict()))

        # Use input that drives mu activations to moderate values (not 0, not huge)
        # so tanh is not at exact saturation but still attenuates vs identity
        T.manual_seed(42)
        x = T.randn(32, INPUT_SIZE, device=tanh_net.device) * 0.5

        # --- tanh network gradient ---
        tanh_net.zero_grad()
        out_tanh = tanh_net(x)
        loss_tanh = out_tanh.sum()
        loss_tanh.backward()
        grad_tanh = tanh_net.mu.weight.grad.abs().mean().item()

        # --- linear network gradient ---
        linear_net.zero_grad()
        out_linear = linear_net(x)
        loss_linear = out_linear.sum()
        loss_linear.backward()
        grad_linear = linear_net.mu.weight.grad.abs().mean().item()

        assert grad_linear >= grad_tanh, (
            f"linear grad ({grad_linear:.6f}) should be >= tanh grad ({grad_tanh:.6f}): "
            "tanh always attenuates gradients (tanh'(x) <= 1)"
        )
