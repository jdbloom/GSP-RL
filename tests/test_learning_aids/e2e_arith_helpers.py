"""Deterministic one-step builders for the GSP_E2E_UNIFIED_TARGET_ARITH gate.

Two learn paths are frozen/exercised:

  * ``learn_DDQN_e2e``          — the ONLY fn with the legacy target-arithmetic
                                  bypass (raw rewards + gamma*bootstrap, no
                                  REWARD_SCALE / Q_TARGET_CLIP / critic grad
                                  clip). The flag routes it through _q_target.
  * ``learn_DDQN_jepa_coupled`` — verified to ALREADY use _q_target /
                                  _critic_loss_fn / _clip_critic_grad (no
                                  bypass); frozen here as insurance that the
                                  flag work never perturbs it.

Same determinism contract as ``golden_helpers.py``: construction + buffer fill
pinned at seeds 0/0, the learn call re-pinned at 123/123 so replay sampling is
independent of how many randoms the fill consumed. Only numpy/torch + gsp_rl
imports (no superrepo dependency).
"""
import numpy as np
import torch as T

from gsp_rl.src.actors.actor import Actor
from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.buffers.replay import ReplayBuffer
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork

# --- learn_DDQN_e2e fixture dimensions (mirrors tests/test_e2e_gsp) ---------
ENV_OBS_SIZE = 31
GSP_OBS_SIZE = 6
GSP_OUTPUT_SIZE = 1
AUGMENTED_OBS_SIZE = ENV_OBS_SIZE + GSP_OUTPUT_SIZE
NUM_ACTIONS = 5
BATCH_SIZE = 16
MEM_SIZE = 200
LR = 1e-3

E2E_BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": LR,
    "BETA": LR,
    "LR": LR,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": BATCH_SIZE,
    "MEM_SIZE": MEM_SIZE,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": BATCH_SIZE,
    "GSP_E2E_ENABLED": True,
}


def make_e2e_setup(extra_config=None, reward_scale_fill=1.0):
    """Build (aids, networks, gsp_networks) with a deterministically filled
    replay. Seeds are pinned at 0/0 by the caller-facing one-step fns; this
    builder consumes RNG, so callers must seed BEFORE calling it.

    reward_scale_fill: multiplier on the stored rewards (used by the
    clip-boundary test to push |target| past Q_TARGET_CLIP with real stored
    data, not by editing tensors post-hoc).
    """
    cfg = dict(E2E_BASE_CONFIG)
    if extra_config:
        cfg.update(extra_config)
    aids = NetworkAids(cfg)
    aids.input_size = ENV_OBS_SIZE  # normally set by Actor

    q_eval = DDQN(id=0, lr=LR, input_size=AUGMENTED_OBS_SIZE,
                  output_size=NUM_ACTIONS, fc1_dims=32, fc2_dims=32)
    q_next = DDQN(id=0, lr=LR, input_size=AUGMENTED_OBS_SIZE,
                  output_size=NUM_ACTIONS, fc1_dims=32, fc2_dims=32)
    replay = ReplayBuffer(
        max_size=MEM_SIZE,
        num_observations=AUGMENTED_OBS_SIZE,
        num_actions=1,
        action_type='Discrete',
        gsp_obs_size=GSP_OBS_SIZE,
    )
    networks = {
        'q_eval': q_eval,
        'q_next': q_next,
        'replay': replay,
        'learning_scheme': 'DDQN',
        'learn_step_counter': 0,
    }
    gsp_actor = DDPGActorNetwork(
        id=0, lr=LR, input_size=GSP_OBS_SIZE, output_size=GSP_OUTPUT_SIZE,
        fc1_dims=32, fc2_dims=16, min_max_action=1.0, use_linear_output=True,
    )
    gsp_networks = {
        'actor': gsp_actor,
        'learning_scheme': 'DDPG',
        'learn_step_counter': 0,
    }

    rng = np.random.default_rng(42)
    for _ in range(MEM_SIZE):
        state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        action = int(rng.integers(0, NUM_ACTIONS))
        reward = float(rng.standard_normal()) * float(reward_scale_fill)
        state_ = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        done = bool(rng.integers(0, 2))
        gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
        gsp_label = rng.standard_normal(1).astype(np.float32)
        replay.store_transition(state, action, reward, state_, done,
                                gsp_obs=gsp_obs, gsp_label=gsp_label)
    return aids, networks, gsp_networks


def params_flat(net):
    return T.cat([p.data.flatten().clone() for p in net.parameters()])


def one_e2e_learn_step(extra_config=None, reward_scale_fill=1.0):
    """One fully deterministic learn_DDQN_e2e step.

    Returns the diagnostics dict plus post-step q_eval / GSP-head param
    snapshots (numpy) for golden comparison.
    """
    np.random.seed(0)
    T.manual_seed(0)
    aids, networks, gsp_networks = make_e2e_setup(
        extra_config, reward_scale_fill=reward_scale_fill)

    np.random.seed(123)
    T.manual_seed(123)
    diag = aids.learn_DDQN_e2e(networks, gsp_networks)

    return {
        'ddqn_loss': float(diag['ddqn_loss']),
        'gsp_mse_loss': float(diag['gsp_mse_loss']),
        'total_loss': float(diag['total_loss']),
        'q_eval_params': params_flat(networks['q_eval']).cpu().numpy(),
        'gsp_actor_params': params_flat(gsp_networks['actor']).cpu().numpy(),
    }


# --- learn_DDQN_jepa_coupled fixture (mirrors tests/test_jepa_coupled.py) ---
JC_ENV_OBS = 4
JC_GSP_INPUT = 6
JC_ENC_DIM = 8
JC_NUM_ACTIONS = 3
JC_BATCH = 16

JC_BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 1e-3,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": JC_BATCH,
    "MEM_SIZE": 2000,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 1,
    "GSP_BATCH_SIZE": JC_BATCH,
    "GSP_JEPA_ENABLED": True,
    "GSP_ENCODER_DIM": JC_ENC_DIM,
    "GSP_JEPA_COUPLE_VALUE": True,
}


def one_jepa_coupled_learn_step(extra_config=None):
    """One fully deterministic learn_DDQN_jepa_coupled step.

    Returns loss scalars plus post-step q_eval / online-encoder param
    snapshots (numpy) for golden comparison.
    """
    np.random.seed(0)
    T.manual_seed(0)
    cfg = dict(JC_BASE_CONFIG)
    if extra_config:
        cfg.update(extra_config)
    actor = Actor(
        id=0,
        config=cfg,
        network="DDQN",
        input_size=JC_ENV_OBS,
        output_size=JC_NUM_ACTIONS,
        min_max_action=1,
        meta_param_size=0,
        gsp=True,
        gsp_input_size=JC_GSP_INPUT,
        gsp_output_size=1,
    )
    aug_size = actor.network_input_size  # JC_ENV_OBS + JC_ENC_DIM
    for _ in range(JC_BATCH * 2):
        s = np.random.randn(aug_size).astype(np.float32)
        s_ = np.random.randn(aug_size).astype(np.float32)
        a = int(np.random.randint(0, JC_NUM_ACTIONS))
        gsp_obs = np.random.randn(JC_GSP_INPUT).astype(np.float32)
        actor.store_agent_transition(
            s, a, 0.5, s_, False, gsp_obs=gsp_obs,
            gsp_label=np.zeros(1, np.float32),
        )

    np.random.seed(123)
    T.manual_seed(123)
    stats = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)

    return {
        'ddqn_loss': float(stats['ddqn_loss']),
        'jepa_pred_mse': float(stats['jepa_pred_mse']),
        'total_loss': float(stats['total_loss']),
        'q_eval_params': params_flat(actor.networks['q_eval']).cpu().numpy(),
        'encoder_params': params_flat(actor.gsp_encoder_online).cpu().numpy(),
    }
