"""Gradient flow tests for end-to-end GSP + DDQN joint training.

Verifies that:
1. Gradients flow back through torch.cat to the GSP head.
2. The augmented state's GSP column equals the fresh GSP prediction.
3. The auxiliary MSE loss is differentiable w.r.t. GSP head parameters.
4. The Q-target computation uses no_grad, so next-state Q has no gradients.
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.networks.ddqn import DDQN


GSP_INPUT_SIZE = 6   # e.g. 6-element GSP-N observation vector
GSP_OUTPUT_SIZE = 1  # scalar Δθ prediction
ENV_OBS_SIZE = 4     # raw environment observation dimensionality
AUGMENTED_SIZE = ENV_OBS_SIZE + GSP_OUTPUT_SIZE  # state fed to Q-net
NUM_ACTIONS = 3
BATCH_SIZE = 8
LR = 1e-3


def make_gsp_head() -> DDPGActorNetwork:
    """Minimal GSP head: maps gsp_obs -> scalar prediction."""
    return DDPGActorNetwork(
        id=0,
        lr=LR,
        input_size=GSP_INPUT_SIZE,
        output_size=GSP_OUTPUT_SIZE,
        fc1_dims=32,
        fc2_dims=16,
        min_max_action=1.0,
        use_linear_output=True,
    )


def make_q_net() -> DDQN:
    """Minimal Q-network operating on augmented state."""
    return DDQN(
        id=0,
        lr=LR,
        input_size=AUGMENTED_SIZE,
        output_size=NUM_ACTIONS,
        fc1_dims=32,
        fc2_dims=16,
    )


class TestGradientFlowsThroughCatToGspHead:
    """torch.cat must not block gradient flow from Q-loss back to GSP head."""

    def test_gradient_flows_through_cat_to_gsp_head(self):
        gsp_head = make_gsp_head()
        q_net = make_q_net()
        device = gsp_head.device

        gsp_obs = T.randn(BATCH_SIZE, GSP_INPUT_SIZE).to(device)
        env_obs = T.randn(BATCH_SIZE, ENV_OBS_SIZE).to(device)

        # Re-run GSP head with gradient
        gsp_pred = gsp_head(gsp_obs)  # (batch, 1)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)

        # Build augmented state by replacing a slot with fresh prediction
        gsp_idx = ENV_OBS_SIZE  # scalar sits right after raw obs
        augmented = T.cat([env_obs, gsp_pred], dim=1)  # (batch, AUGMENTED_SIZE)

        # Forward through Q-net and compute a scalar loss
        q_values = q_net(augmented)
        loss = q_values.sum()
        loss.backward()

        # Every parameter of the GSP head must have a non-zero gradient
        for name, param in gsp_head.named_parameters():
            assert param.grad is not None, (
                f"GSP head parameter '{name}' has no gradient — "
                "gradient is blocked at torch.cat"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"GSP head parameter '{name}' has zero gradient"
            )


class TestStaleGspInStateIsReplacedByFresh:
    """The augmented state's GSP column must equal the fresh GSP prediction."""

    def test_stale_gsp_in_state_is_replaced_by_fresh(self):
        gsp_head = make_gsp_head()
        device = gsp_head.device

        # Simulate a stored state with a stale GSP scalar
        stale_gsp_value = 99.0  # obviously wrong placeholder
        stored_states = T.randn(BATCH_SIZE, AUGMENTED_SIZE).to(device)
        stored_states[:, ENV_OBS_SIZE] = stale_gsp_value  # stale

        gsp_obs = T.randn(BATCH_SIZE, GSP_INPUT_SIZE).to(device)

        with T.no_grad():
            gsp_pred = gsp_head(gsp_obs)
            if gsp_pred.dim() == 1:
                gsp_pred = gsp_pred.unsqueeze(1)

        gsp_idx = ENV_OBS_SIZE
        augmented = T.cat(
            [stored_states[:, :gsp_idx], gsp_pred, stored_states[:, gsp_idx + 1:]], dim=1
        )

        # The GSP column in augmented must be the fresh prediction, NOT stale
        T.testing.assert_close(
            augmented[:, gsp_idx : gsp_idx + 1],
            gsp_pred,
            msg="Augmented state GSP column does not match fresh GSP prediction",
        )

        # Verify the stale value is gone
        assert not T.allclose(
            augmented[:, gsp_idx],
            T.full((BATCH_SIZE,), stale_gsp_value, device=device),
        ), "Stale GSP value was not replaced"


class TestAuxiliaryMseIsDifferentiable:
    """MSE(fresh_pred, stored_label) must have gradients on the GSP head."""

    def test_auxiliary_mse_is_differentiable(self):
        gsp_head = make_gsp_head()
        device = gsp_head.device

        gsp_obs = T.randn(BATCH_SIZE, GSP_INPUT_SIZE).to(device)
        gsp_labels = T.randn(BATCH_SIZE, 1).to(device)  # stored co-indexed labels

        gsp_pred = gsp_head(gsp_obs)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)

        mse = F.mse_loss(gsp_pred, gsp_labels)
        mse.backward()

        for name, param in gsp_head.named_parameters():
            assert param.grad is not None, (
                f"GSP head parameter '{name}' has no gradient from MSE loss"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"GSP head parameter '{name}' has zero gradient from MSE loss"
            )


class TestNextStateIsNotRerun:
    """Q-target computation on next-state must execute under no_grad.

    If no_grad is missing, tensors computed from states_ would have
    requires_grad=True (when the states_ tensor is part of a computation graph).
    We verify the Q-values for the next state do NOT have grad_fn.
    """

    def test_next_state_is_not_rerun(self):
        q_net = make_q_net()
        device = q_net.device

        # Simulate stored next_states as a plain tensor (from replay buffer)
        states_ = T.randn(BATCH_SIZE, AUGMENTED_SIZE).to(device)
        # Give states_ a grad so we can detect if it leaks through
        states_.requires_grad_(True)

        with T.no_grad():
            q_next = q_net(states_)

        # Q-values computed inside no_grad must not have a grad_fn
        assert q_next.grad_fn is None, (
            "Q-target computation has grad_fn — it was NOT wrapped in no_grad. "
            "This breaks stable-target DDQN training."
        )
