"""Regression for the K>1 (delta_theta_traj) E2E path (confirmed 2026-07-08).

Two coupled K-inconsistencies used to crash the size-K trajectory E2E head:

  Bug B — actor-width splice. learn_DDQN_e2e / learn_TD3_e2e splice the freshly
  re-run head prediction into the stored augmented state. The GSP slot is
  gsp_network_output (K) wide, but the splice used a hardcoded width-1 slot
  (states[:, gsp_idx + 1:]), so for K=5 it left K-1 stale columns in place and
  grew the augmented width past the Q-net input, crashing the forward with
  `mat1 and mat2 shapes cannot be multiplied (64x40 and 36x64)`.

  Bug C — E2E label width. The main-replay gsp_label column was hardcoded width
  1, so a size-K head prediction could not be regressed against it
  (`gsp_labels.view_as(gsp_pred)` failed). ReplayBuffer now takes a
  gsp_label_size (default 1, byte-identical legacy); the host passes K.

These tests drive the real learn step with a size-K head + a K-wide label column
and assert the whole step runs and the spliced widths / scaling are correct.
"""
import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.buffers.replay import ReplayBuffer


ENV_OBS_SIZE = 31
GSP_OBS_SIZE = 19
NUM_ACTIONS = 5
BATCH_SIZE = 16
MEM_SIZE = 200
LR = 1e-3

MINIMAL_CONFIG = {
    "GAMMA": 0.99, "TAU": 0.005, "ALPHA": LR, "BETA": LR, "LR": LR,
    "EPSILON": 1.0, "EPS_MIN": 0.01, "EPS_DEC": 0.001,
    "BATCH_SIZE": BATCH_SIZE, "MEM_SIZE": MEM_SIZE,
    "REPLACE_TARGET_COUNTER": 100, "NOISE": 0.1, "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0, "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": BATCH_SIZE,
}


def _make_aids(K):
    aids = NetworkAids(MINIMAL_CONFIG)
    aids.input_size = ENV_OBS_SIZE
    aids.gsp_network_output = K   # normally set by Actor; the splice reads it
    return aids


def _make_ddqn_networks(K):
    augmented = ENV_OBS_SIZE + K
    q_eval = DDQN(id=0, lr=LR, input_size=augmented, output_size=NUM_ACTIONS,
                  fc1_dims=32, fc2_dims=32)
    q_next = DDQN(id=0, lr=LR, input_size=augmented, output_size=NUM_ACTIONS,
                  fc1_dims=32, fc2_dims=32)
    replay = ReplayBuffer(
        max_size=MEM_SIZE, num_observations=augmented, num_actions=1,
        action_type='Discrete', gsp_obs_size=GSP_OBS_SIZE, gsp_label_size=K,
    )
    return {'q_eval': q_eval, 'q_next': q_next, 'replay': replay,
            'learning_scheme': 'DDQN', 'learn_step_counter': 0}


def _make_gsp_networks(K):
    actor = DDPGActorNetwork(
        id=0, lr=LR, input_size=GSP_OBS_SIZE, output_size=K,
        fc1_dims=32, fc2_dims=16, min_max_action=1.0, use_linear_output=True,
    )
    return {'actor': actor, 'learning_scheme': 'DDPG', 'learn_step_counter': 0}


def _fill_replay(replay, K, n):
    augmented = ENV_OBS_SIZE + K
    rng = np.random.default_rng(7)
    for _ in range(n):
        state = rng.standard_normal(augmented).astype(np.float32)
        action = int(rng.integers(0, NUM_ACTIONS))
        reward = float(rng.standard_normal())
        state_ = rng.standard_normal(augmented).astype(np.float32)
        done = bool(rng.integers(0, 2))
        gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
        gsp_label = rng.standard_normal(K).astype(np.float32)  # size-K E2E label
        replay.store_transition(state, action, reward, state_, done,
                                gsp_obs=gsp_obs, gsp_label=gsp_label)


def _setup(K):
    aids = _make_aids(K)
    networks = _make_ddqn_networks(K)
    gsp_networks = _make_gsp_networks(K)
    _fill_replay(networks['replay'], K, MEM_SIZE)
    return aids, networks, gsp_networks


# --- Bug C: K-wide label buffer ---------------------------------------------

@pytest.mark.parametrize("K", [1, 3, 5])
def test_replay_gsp_label_column_is_k_wide(K):
    replay = ReplayBuffer(
        max_size=10, num_observations=ENV_OBS_SIZE + K, num_actions=1,
        action_type='Discrete', gsp_obs_size=GSP_OBS_SIZE, gsp_label_size=K,
    )
    assert replay.gsp_label_memory.shape == (10, K)
    label = np.arange(K, dtype=np.float32)
    replay.store_transition(
        np.zeros(ENV_OBS_SIZE + K, dtype=np.float32), 0, 0.0,
        np.zeros(ENV_OBS_SIZE + K, dtype=np.float32), False,
        gsp_obs=np.zeros(GSP_OBS_SIZE, dtype=np.float32), gsp_label=label,
    )
    np.testing.assert_array_equal(replay.gsp_label_memory[0], label)


def test_replay_gsp_label_size_defaults_to_one():
    """Default gsp_label_size=1 keeps the legacy scalar column byte-identical."""
    replay = ReplayBuffer(
        max_size=10, num_observations=ENV_OBS_SIZE + 1, num_actions=1,
        action_type='Discrete', gsp_obs_size=GSP_OBS_SIZE,
    )
    assert replay.gsp_label_memory.shape == (10, 1)
    assert replay._zero_gsp_label.shape == (1,)


# --- Bug B: K-aware splice ---------------------------------------------------

@pytest.mark.parametrize("K", [1, 3, 5])
def test_learn_ddqn_e2e_runs_end_to_end_for_kdim(K):
    """The full E2E learn step must complete (splice + view_as + MSE) for K>1."""
    aids, networks, gsp_networks = _setup(K)
    result = aids.learn_DDQN_e2e(networks, gsp_networks)  # must not raise
    assert np.isfinite(result['total_loss'])
    assert np.isfinite(result['gsp_mse_loss'])


@pytest.mark.parametrize("K", [1, 5])
def test_augmented_width_matches_qnet_input(K):
    """The augmented state fed to q_eval must be exactly input_size + K wide —
    the width the Q-net was built for. This is the invariant `64x40 and 36x64`
    violated."""
    aids, networks, gsp_networks = _setup(K)
    cap = {}
    q_fwd = networks['q_eval'].forward

    def q_spy(x):
        cap.setdefault('aug_width', x.shape[1])
        return q_fwd(x)

    networks['q_eval'].forward = q_spy
    aids.learn_DDQN_e2e(networks, gsp_networks)
    assert cap['aug_width'] == ENV_OBS_SIZE + K, (
        f"K={K}: augmented width {cap['aug_width']} != {ENV_OBS_SIZE + K}"
    )


@pytest.mark.parametrize("K", [3, 5])
def test_kdim_actor_slot_is_raw_not_scaled(K):
    """For K>1 the spliced actor slot must be the RAW size-K head output
    (make_agent_state does NOT rescale the vector path)."""
    aids, networks, gsp_networks = _setup(K)
    cap = {}
    head_fwd = gsp_networks['actor'].forward
    q_fwd = networks['q_eval'].forward

    def head_spy(x):
        out = head_fwd(x)
        cap.setdefault('pred', out.detach().clone())
        return out

    def q_spy(x):
        cap.setdefault('aug', x.detach().clone())
        return q_fwd(x)

    gsp_networks['actor'].forward = head_spy
    networks['q_eval'].forward = q_spy
    aids.learn_DDQN_e2e(networks, gsp_networks)

    pred = cap['pred'].reshape(-1, K)
    slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + K]
    assert T.allclose(slot, pred, atol=1e-5), (
        f"K={K}: actor GSP slot must be the RAW size-{K} head output"
    )
    scale = float(np.degrees(1.0) / 10.0)
    assert not T.allclose(slot, pred * scale, atol=1e-4), (
        f"K={K}: actor slot is scaled — scalar-only degrees/10 leaked into the "
        "K>1 vector path"
    )


def test_k1_actor_slot_keeps_legacy_scaling():
    """K=1 must keep the historical degrees/10 scaling — byte-identical legacy."""
    K = 1
    aids, networks, gsp_networks = _setup(K)
    cap = {}
    head_fwd = gsp_networks['actor'].forward
    q_fwd = networks['q_eval'].forward

    def head_spy(x):
        out = head_fwd(x)
        cap.setdefault('pred', out.detach().clone())
        return out

    def q_spy(x):
        cap.setdefault('aug', x.detach().clone())
        return q_fwd(x)

    gsp_networks['actor'].forward = head_spy
    networks['q_eval'].forward = q_spy
    aids.learn_DDQN_e2e(networks, gsp_networks)

    scale = float(np.degrees(1.0) / 10.0)
    pred = cap['pred'].reshape(-1, 1)
    slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + 1]
    assert T.allclose(slot, pred * scale, atol=1e-5), (
        "K=1 actor slot must stay the scaled (degrees/10) head output"
    )
