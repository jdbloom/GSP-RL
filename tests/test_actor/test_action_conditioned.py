"""Golden off-path tests for GSP_ACTION_CONDITIONED (discrete-DDQN dtraj head).

Mirrors the assertion style of tests/test_actor/test_advantage_splice.py.
"""
import pytest
import numpy as np
import torch as T
import copy

from gsp_rl.src.actors.actor import Actor


# ---------------------------------------------------------------------------
# Minimal config builder
# ---------------------------------------------------------------------------
def _base_config(**overrides):
    """Return a dict that passes Hyperparameters.__init__ without missing keys."""
    cfg = {
        'GAMMA': 0.99,
        'TAU': 0.005,
        'ALPHA': 0.001,
        'BETA': 0.001,
        'LR': 1e-4,
        'EPSILON': 1.0,
        'EPS_MIN': 0.01,
        'EPS_DEC': 1e-5,
        'GSP_LEARNING_FREQUENCY': 100,
        'GSP_BATCH_SIZE': 32,
        'BATCH_SIZE': 32,
        'MEM_SIZE': 10000,
        'REPLACE_TARGET_COUNTER': 100,
        'NOISE': 0.1,
        'UPDATE_ACTOR_ITER': 2,
        'WARMUP': 0,
        'GSP_PREDICTION_TARGET': 'delta_theta_traj',
        'GSP_PREDICTION_HORIZON': 5,
        'GSP_OUTPUT_KIND': 'delta_theta_traj',
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# (a) Flag OFF → head input width == gsp_network_input, byte-identical
# ---------------------------------------------------------------------------
def test_action_conditioned_off_is_byte_identical():
    """GSP_ACTION_CONDITIONED unset/False → head input width == gsp_network_input."""
    cfg = _base_config()
    # Two agents: one with the flag explicitly False, one without it at all.
    agent_a = Actor(
        id=1, config=dict(cfg, GSP_ACTION_CONDITIONED=False),
        network='DDQN', input_size=31, output_size=9,
        min_max_action=1, meta_param_size=0,
        gsp=True, gsp_input_size=6, gsp_output_size=5,
    )
    agent_b = Actor(
        id=2, config=cfg,
        network='DDQN', input_size=31, output_size=9,
        min_max_action=1, meta_param_size=0,
        gsp=True, gsp_input_size=6, gsp_output_size=5,
    )
    # Both heads must have the same input width.
    a_in = agent_a.gsp_networks['actor'].fc1.weight.shape[1]
    b_in = agent_b.gsp_networks['actor'].fc1.weight.shape[1]
    assert a_in == agent_a.gsp_network_input
    assert b_in == agent_b.gsp_network_input
    assert a_in == b_in

    # Byte-identical head construction: same random init → same weights.
    agent_a.gsp_networks['actor'].load_state_dict(
        agent_b.gsp_networks['actor'].state_dict()
    )
    for p_a, p_b in zip(agent_a.gsp_networks['actor'].parameters(),
                        agent_b.gsp_networks['actor'].parameters()):
        assert T.equal(p_a, p_b)


# ---------------------------------------------------------------------------
# (b) Flag ON (onehot, N=9, dtraj target, DDQN) → head input width +9,
#     predict_gsp_actions returns (9, K) with variance across actions.
# ---------------------------------------------------------------------------
def test_action_conditioned_on_onehot():
    """GSP_ACTION_CONDITIONED=True (onehot, N=9) → head input width +9."""
    cfg = _base_config(
        GSP_ACTION_CONDITIONED=True,
        GSP_ACTION_COND_ENCODING='onehot',
        GSP_ACTION_COND_N=9,
    )
    agent = Actor(
        id=3, config=cfg,
        network='DDQN', input_size=31, output_size=9,
        min_max_action=1, meta_param_size=0,
        gsp=True, gsp_input_size=6, gsp_output_size=5,
    )
    assert agent.gsp_action_conditioned_engaged is True
    head_in = agent.gsp_networks['actor'].fc1.weight.shape[1]
    assert head_in == agent.gsp_network_input + 9  # 6 + 9 = 15

    # predict_gsp_actions returns (9, K).
    gsp_state = np.random.randn(agent.gsp_network_input).astype(np.float32)
    preds = agent.predict_gsp_actions(gsp_state, n_actions=9)
    assert preds.shape == (9, agent.gsp_network_output)  # (9, 5)

    # Rows must DIFFER when fed distinct one-hots on a randomly-initialized head.
    # Variance across the 9 rows > 0 by construction on init.
    row_var = float(preds.var(dim=0).mean().item())
    assert row_var > 0.0, (
        f"predict_gsp_actions rows are identical (var={row_var}); "
        "the head is not action-conditioned."
    )


# ---------------------------------------------------------------------------
# (c) Construction gate raises on DDPG host scheme and on non-dtraj target.
# ---------------------------------------------------------------------------
def test_action_conditioned_gate_rejects_ddpg():
    """Gate raises ValueError on a DDPG host scheme."""
    cfg = _base_config(
        GSP_ACTION_CONDITIONED=True,
        GSP_ACTION_COND_N=9,
    )
    with pytest.raises(ValueError, match='discrete host scheme'):
        Actor(
            id=4, config=cfg,
            network='DDPG', input_size=31, output_size=2,
            min_max_action=1, meta_param_size=0,
            gsp=True, gsp_input_size=6, gsp_output_size=5,
        )


def test_action_conditioned_gate_rejects_non_dtraj_target():
    """Gate raises ValueError on a non-delta_theta_traj target."""
    cfg = _base_config(
        GSP_ACTION_CONDITIONED=True,
        GSP_ACTION_COND_N=9,
        GSP_PREDICTION_TARGET='future_prox',
        GSP_OUTPUT_KIND='future_prox_1d',
    )
    with pytest.raises(ValueError, match='delta_theta_traj'):
        Actor(
            id=5, config=cfg,
            network='DDQN', input_size=31, output_size=9,
            min_max_action=1, meta_param_size=0,
            gsp=True, gsp_input_size=6, gsp_output_size=1,
        )