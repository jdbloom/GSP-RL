"""Per-network DIAGNOSTIC_PROFILE tests.

Verifies that every network class carrying a DIAGNOSTIC_PROFILE produces the
expected set of diagnostic keys when passed through the diagnostic dispatch
functions, and that no returned finite values are NaN/Inf.

One test per network class:
- DQN
- DDQN
- DDPGActorNetwork
- DDPGCriticNetwork (weight norms only — critic forward takes state+action)
- TD3ActorNetwork
- TD3CriticNetwork (weight norms only)
- RDDPGActorNetwork
- RDDPGCriticNetwork (weight norms only)
- EnvironmentEncoder (LSTM; hidden-norm path)
- AttentionEncoder (attention-entropy path)

Design: each test constructs the network with small realistic dims, builds a
small eval batch, runs the relevant diagnostic functions directly (not via
Actor), and asserts:
1. The expected key set is present in the returned dict.
2. No finite values are NaN or Inf (attention_entropy may return NaN if the
   hook cannot be installed — skip the assertion in that case).
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from gsp_rl.src.networks.dqn import DQN
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork, DDPGCriticNetwork
from gsp_rl.src.networks.td3 import TD3ActorNetwork, TD3CriticNetwork
from gsp_rl.src.networks.rddpg import RDDPGActorNetwork, RDDPGCriticNetwork
from gsp_rl.src.networks.lstm import EnvironmentEncoder
from gsp_rl.src.networks.self_attention import AttentionEncoder

from gsp_rl.src.actors.diagnostics import (
    compute_fau,
    compute_overactive_fau,
    compute_weight_norms,
    compute_effective_rank,
    compute_q_action_gap,
    compute_hidden_norm,
    compute_attention_entropy,
)

# Small sizes to keep tests fast
INPUT_SIZE = 8
OUTPUT_SIZE = 4
BATCH_SIZE = 32

# ---------------------------------------------------------------------------------
# Helper: run diagnostics given a profile and a network, return merged dict
# ---------------------------------------------------------------------------------

def _run_profile_diagnostics(net, batch, profile):
    """Run all diagnostics dictated by ``profile`` and return merged result dict."""
    out = {}
    fau_layers = profile.get('fau_layers', [])
    wnorm_layers = profile.get('wnorm_layers', [])
    has_penultimate = profile.get('has_penultimate', False)
    output_kind = profile.get('output_kind', '')

    if fau_layers:
        out.update(compute_fau(net, batch, fau_layers))
        out.update(compute_overactive_fau(net, batch, fau_layers))

    if wnorm_layers:
        out.update(compute_weight_norms(net, wnorm_layers))

    if has_penultimate and hasattr(net, 'penultimate'):
        out['erank_penult'] = compute_effective_rank(net, batch, 'penultimate')

    if output_kind == 'q_values':
        out.update(compute_q_action_gap(net, batch))

    if output_kind == 'lstm_hidden':
        out['hidden_norm'] = compute_hidden_norm(net, batch)

    if output_kind == 'attention':
        out['attention_entropy'] = compute_attention_entropy(net, batch)

    return out


def _assert_no_nan_inf(result: dict, skip_keys: set | None = None):
    """Assert that all finite values in ``result`` are not NaN/Inf."""
    skip_keys = skip_keys or set()
    for k, v in result.items():
        if k in skip_keys:
            continue
        assert math.isfinite(v), f"Non-finite value for key '{k}': {v}"


# ---------------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------------

def test_dqn_diagnostic_profile_keys_and_values():
    """DQN: FAU + weight norms + effective rank + Q-gap all present and finite."""
    net = DQN(id=1, lr=1e-4, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
    profile = DQN.DIAGNOSTIC_PROFILE
    batch = torch.randn(BATCH_SIZE, INPUT_SIZE).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    # FAU keys
    assert 'fau_fc1' in result
    assert 'fau_fc2' in result
    assert 'overactive_fc1' in result
    assert 'overactive_fc2' in result
    # Weight norm keys
    assert 'wnorm_fc1' in result
    assert 'wnorm_fc2' in result
    assert 'wnorm_fc3' in result
    # Effective rank (has_penultimate=True)
    assert 'erank_penult' in result
    # Q-action gap (output_kind='q_values')
    assert 'q_action_gap_mean' in result
    assert 'q_action_gap_std' in result
    assert 'q_max_mean' in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DDQN
# ---------------------------------------------------------------------------------

def test_ddqn_diagnostic_profile_keys_and_values():
    """DDQN: same profile as DQN — full diagnostic coverage."""
    net = DDQN(id=1, lr=1e-4, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
    profile = DDQN.DIAGNOSTIC_PROFILE
    batch = torch.randn(BATCH_SIZE, INPUT_SIZE).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    assert 'fau_fc1' in result
    assert 'fau_fc2' in result
    assert 'wnorm_fc1' in result
    assert 'wnorm_fc3' in result
    assert 'erank_penult' in result
    assert 'q_action_gap_mean' in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DDPGActorNetwork
# ---------------------------------------------------------------------------------

def test_ddpg_actor_diagnostic_profile_keys_and_values():
    """DDPGActorNetwork: FAU + weight norms + effective rank; no Q-gap."""
    net = DDPGActorNetwork(id=1, lr=1e-4, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
    profile = DDPGActorNetwork.DIAGNOSTIC_PROFILE
    batch = torch.randn(BATCH_SIZE, INPUT_SIZE).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    assert 'fau_fc1' in result
    assert 'fau_fc2' in result
    assert 'wnorm_fc1' in result
    assert 'wnorm_fc2' in result
    assert 'wnorm_mu' in result
    assert 'erank_penult' in result
    # No Q-action gap for continuous action
    assert 'q_action_gap_mean' not in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# DDPGCriticNetwork
# ---------------------------------------------------------------------------------

def test_ddpg_critic_diagnostic_profile_weight_norms():
    """DDPGCriticNetwork: weight norms only (forward takes state+action pairs)."""
    # Critic input_size = state_dim + action_dim
    net = DDPGCriticNetwork(
        id=1, lr=1e-4, input_size=INPUT_SIZE + OUTPUT_SIZE, output_size=1
    )
    profile = DDPGCriticNetwork.DIAGNOSTIC_PROFILE
    # Only weight norms are run for critics in Actor._diagnose_network (critic path)
    result = compute_weight_norms(net, profile['wnorm_layers'])

    assert 'wnorm_fc1' in result
    assert 'wnorm_fc2' in result
    assert 'wnorm_q' in result
    # has_penultimate=False and output_kind='q_scalar' — no erank, no q_action_gap
    assert profile['has_penultimate'] is False
    assert profile['output_kind'] == 'q_scalar'

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# TD3ActorNetwork
# ---------------------------------------------------------------------------------

def test_td3_actor_diagnostic_profile_keys_and_values():
    """TD3ActorNetwork: FAU + weight norms + effective rank (via new penultimate)."""
    net = TD3ActorNetwork(
        id=1, alpha=1e-4, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE,
        fc1_dims=64, fc2_dims=32
    )
    profile = TD3ActorNetwork.DIAGNOSTIC_PROFILE
    batch = torch.randn(BATCH_SIZE, INPUT_SIZE).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    assert 'fau_fc1' in result
    assert 'fau_fc2' in result
    assert 'wnorm_fc1' in result
    assert 'wnorm_fc2' in result
    assert 'wnorm_mu' in result
    assert 'erank_penult' in result
    # No Q-action gap for continuous action
    assert 'q_action_gap_mean' not in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# TD3CriticNetwork
# ---------------------------------------------------------------------------------

def test_td3_critic_diagnostic_profile_weight_norms():
    """TD3CriticNetwork: weight norms only — single critic diagnosed (critic_1)."""
    net = TD3CriticNetwork(
        id=1, beta=1e-4, input_size=INPUT_SIZE + OUTPUT_SIZE, output_size=OUTPUT_SIZE,
        fc1_dims=64, fc2_dims=32
    )
    profile = TD3CriticNetwork.DIAGNOSTIC_PROFILE
    result = compute_weight_norms(net, profile['wnorm_layers'])

    assert 'wnorm_fc1' in result
    assert 'wnorm_fc2' in result
    assert 'wnorm_q1' in result
    assert profile['has_penultimate'] is False
    assert profile['output_kind'] == 'q_scalar'

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# RDDPGActorNetwork
# ---------------------------------------------------------------------------------

def test_rddpg_actor_diagnostic_profile_keys_and_values():
    """RDDPGActorNetwork: FAU on inner DDPG fc layers + effective rank via penultimate."""
    lstm_input = INPUT_SIZE
    lstm_output = 16  # meta_param_size (encoding size)
    action_size = OUTPUT_SIZE

    encoder = EnvironmentEncoder(
        input_size=lstm_input,
        output_size=lstm_output,
        hidden_size=32,
        embedding_size=16,
        batch_size=BATCH_SIZE,
        num_layers=2,
        lr=1e-4,
    )
    inner_actor = DDPGActorNetwork(
        id=1, lr=1e-4, input_size=lstm_output, output_size=action_size,
        fc1_dims=32, fc2_dims=16
    )
    net = RDDPGActorNetwork(encoder, inner_actor)

    profile = RDDPGActorNetwork.DIAGNOSTIC_PROFILE
    # eval_batch is (N, raw_obs_dim)
    batch = torch.randn(BATCH_SIZE, lstm_input).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    # FAU on inner actor layers (dotted keys become underscore-joined)
    assert 'fau_actor_fc1' in result
    assert 'fau_actor_fc2' in result
    assert 'overactive_actor_fc1' in result
    # Weight norms on inner actor
    assert 'wnorm_actor_fc1' in result
    assert 'wnorm_actor_fc2' in result
    assert 'wnorm_actor_mu' in result
    # Effective rank via penultimate
    assert 'erank_penult' in result
    # No Q-gap
    assert 'q_action_gap_mean' not in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# RDDPGCriticNetwork
# ---------------------------------------------------------------------------------

def test_rddpg_critic_diagnostic_profile_weight_norms():
    """RDDPGCriticNetwork: weight norms on inner DDPG critic layers."""
    lstm_input = INPUT_SIZE
    lstm_output = 16
    action_size = OUTPUT_SIZE

    encoder = EnvironmentEncoder(
        input_size=lstm_input,
        output_size=lstm_output,
        hidden_size=32,
        embedding_size=16,
        batch_size=BATCH_SIZE,
        num_layers=2,
        lr=1e-4,
    )
    inner_critic = DDPGCriticNetwork(
        id=1, lr=1e-4, input_size=lstm_output + action_size, output_size=1,
        fc1_dims=32, fc2_dims=16
    )
    net = RDDPGCriticNetwork(encoder, inner_critic)
    profile = RDDPGCriticNetwork.DIAGNOSTIC_PROFILE

    result = compute_weight_norms(net, profile['wnorm_layers'])

    assert 'wnorm_critic_fc1' in result
    assert 'wnorm_critic_fc2' in result
    assert 'wnorm_critic_q' in result
    assert profile['has_penultimate'] is False
    assert profile['output_kind'] == 'q_scalar'

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# EnvironmentEncoder (LSTM)
# ---------------------------------------------------------------------------------

def test_environment_encoder_diagnostic_profile_keys():
    """EnvironmentEncoder: no FAU (LSTM gates), LSTM weight norms + hidden norm."""
    net = EnvironmentEncoder(
        input_size=INPUT_SIZE,
        output_size=16,
        hidden_size=32,
        embedding_size=16,
        batch_size=BATCH_SIZE,
        num_layers=2,
        lr=1e-4,
    )
    profile = EnvironmentEncoder.DIAGNOSTIC_PROFILE
    # eval_batch treated as single-step sequences
    batch = torch.randn(BATCH_SIZE, INPUT_SIZE).to(net.device)

    assert profile['fau_layers'] == []  # No ReLU units
    assert profile['has_penultimate'] is False

    result = _run_profile_diagnostics(net, batch, profile)

    # Weight norms on LSTM parameter tensors
    assert 'wnorm_ee_weight_ih_l0' in result
    assert 'wnorm_ee_weight_hh_l0' in result
    # Hidden norm (lstm_hidden output kind)
    assert 'hidden_norm' in result

    _assert_no_nan_inf(result)


# ---------------------------------------------------------------------------------
# AttentionEncoder
# ---------------------------------------------------------------------------------

def test_attention_encoder_diagnostic_profile_keys():
    """AttentionEncoder: weight norm on fc_out; attention_entropy (may be NaN if
    hook cannot be installed, which is acceptable per the stub contract)."""
    SEQ_LEN = 5
    net = AttentionEncoder(
        input_size=INPUT_SIZE,
        output_size=1,
        min_max_action=1.0,
        encode_size=2,
        embed_size=16,
        hidden_size=8,
        heads=2,
        forward_expansion=2,
        dropout=0.0,
        max_length=SEQ_LEN,
    )
    profile = AttentionEncoder.DIAGNOSTIC_PROFILE
    # AttentionEncoder expects (N, seq_len, obs_dim) input
    batch = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE).to(net.device)

    result = _run_profile_diagnostics(net, batch, profile)

    assert 'wnorm_fc_out' in result

    # Attention entropy is computed; assert key present. Value may be NaN only
    # if the softmax hook is unreachable — on the standard AttentionEncoder it
    # should be reachable. We allow NaN as per the stub contract.
    assert 'attention_entropy' in result

    # Weight norm must be finite
    assert math.isfinite(result['wnorm_fc_out'])
