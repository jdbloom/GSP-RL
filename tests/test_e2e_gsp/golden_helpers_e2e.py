"""Self-contained helpers for the T1R golden-equivalence gate.

Freezes the CURRENT (baseline) behavior of the ``learn_DDQN_e2e`` and
``learn_DDQN_jepa_coupled`` learn steps so the T1-residual INERT optimization
(cached ``T.arange`` indices + ``actions.long()`` replacing
``T.LongTensor(np.arange(...))`` / ``actions.type(T.LongTensor)``) can be
proven equivalent.

Construction idiom is copied verbatim from ``tests/test_e2e_gsp/
test_e2e_learn_step.py`` (e2e) and ``tests/test_jepa_coupled.py``
(jepa-coupled) so the exercised code paths match the real tests. Only numpy +
torch + gsp_rl are imported; no dependency on the parent superrepo.

Determinism contract (same as tests/test_learning_aids/golden_helpers.py):
construction + buffer fill pinned with seeds 0/0; the learn call re-pinned
with 123/123 immediately before it so replay sampling and any torch RNG draws
are independent of how many randoms construction consumed.
"""
import numpy as np
import torch as T

from gsp_rl.src.actors.actor import Actor
from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.buffers.replay import ReplayBuffer

# --- e2e dimensions (verbatim from test_e2e_learn_step.py) ---
ENV_OBS_SIZE = 31
GSP_OBS_SIZE = 6
GSP_OUTPUT_SIZE = 1
AUGMENTED_OBS_SIZE = ENV_OBS_SIZE + GSP_OUTPUT_SIZE
NUM_ACTIONS = 5
BATCH_SIZE = 16
MEM_SIZE = 200
LR = 1e-3

E2E_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": LR,
    "BETA": LR,
    "LR": LR,
    "EPSILON": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DEC": 0.001,
    "BATCH_SIZE": BATCH_SIZE,
    "MEM_SIZE": MEM_SIZE,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.1,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": BATCH_SIZE,
}

# --- jepa-coupled dimensions (verbatim from test_jepa_coupled.py) ---
J_ENV_OBS = 4
J_GSP_INPUT = 6
J_ENC_DIM = 8
J_NUM_ACTIONS = 3
J_BATCH = 16


def _params_flat(module: T.nn.Module) -> np.ndarray:
    return T.cat([p.data.flatten().clone() for p in module.parameters()]) \
        .detach().cpu().numpy()


def _make_e2e_setup():
    aids = NetworkAids(E2E_CONFIG)
    aids.input_size = ENV_OBS_SIZE
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
    for _ in range(BATCH_SIZE * 4):
        state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        action = int(rng.integers(0, NUM_ACTIONS))
        reward = float(rng.standard_normal())
        state_ = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        done = bool(rng.integers(0, 2))
        gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
        gsp_label = rng.standard_normal(1).astype(np.float32)
        replay.store_transition(state, action, reward, state_, done,
                                gsp_obs=gsp_obs, gsp_label=gsp_label)
    return aids, networks, gsp_networks


def one_e2e_learn_step() -> dict:
    """One fully-deterministic learn_DDQN_e2e step: losses + post-step params."""
    np.random.seed(0)
    T.manual_seed(0)
    aids, networks, gsp_networks = _make_e2e_setup()

    np.random.seed(123)
    T.manual_seed(123)
    diag = aids.learn_DDQN_e2e(networks, gsp_networks)

    return {
        "ddqn_loss": float(diag["ddqn_loss"]),
        "gsp_mse_loss": float(diag["gsp_mse_loss"]),
        "total_loss": float(diag["total_loss"]),
        "q_eval_params": _params_flat(networks["q_eval"]),
        "gsp_actor_params": _params_flat(gsp_networks["actor"]),
    }


def _make_jepa_actor():
    cfg = {
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.002,
        "LR": 1e-3,
        "EPSILON": 0.0,
        "EPS_MIN": 0.0,
        "EPS_DEC": 0.0,
        "BATCH_SIZE": J_BATCH,
        "MEM_SIZE": 2000,
        "REPLACE_TARGET_COUNTER": 100,
        "NOISE": 0.0,
        "UPDATE_ACTOR_ITER": 2,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 1,
        "GSP_BATCH_SIZE": J_BATCH,
        "GSP_JEPA_ENABLED": True,
        "GSP_ENCODER_DIM": J_ENC_DIM,
        "GSP_JEPA_COUPLE_VALUE": True,
    }
    actor = Actor(
        id=0,
        config=cfg,
        network="DDQN",
        input_size=J_ENV_OBS,
        output_size=J_NUM_ACTIONS,
        min_max_action=1,
        meta_param_size=0,
        gsp=True,
        gsp_input_size=J_GSP_INPUT,
        gsp_output_size=1,
    )
    return actor


def one_jepa_coupled_learn_step() -> dict:
    """One fully-deterministic learn_DDQN_jepa_coupled step."""
    np.random.seed(0)
    T.manual_seed(0)
    actor = _make_jepa_actor()
    aug_size = actor.network_input_size
    for _ in range(J_BATCH * 2):
        s = np.random.randn(aug_size).astype(np.float32)
        s_ = np.random.randn(aug_size).astype(np.float32)
        a = np.random.randint(0, J_NUM_ACTIONS)
        gsp_obs = np.random.randn(J_GSP_INPUT).astype(np.float32)
        actor.store_agent_transition(
            s, a, 0.5, s_, False, gsp_obs=gsp_obs,
            gsp_label=np.zeros(1, np.float32),
        )

    np.random.seed(123)
    T.manual_seed(123)
    diag = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)

    return {
        "ddqn_loss": float(diag["ddqn_loss"]),
        "jepa_pred_mse": float(diag["jepa_pred_mse"]),
        "total_loss": float(diag["total_loss"]),
        "q_eval_params": _params_flat(actor.networks["q_eval"]),
        "encoder_params": _params_flat(actor.gsp_encoder_online),
    }
