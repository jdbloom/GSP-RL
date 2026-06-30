"""Self-contained helpers for the T1 golden-equivalence gate.

These freeze the CURRENT (baseline) behavior of the DQN/DDQN learn steps so a
later INERT (behavior-preserving) optimization can be proven equivalent.

Construction idiom (CONFIG, INPUT_SIZE, output sizes) is copied verbatim from
``tests/test_learning_aids/test_base_learn.py`` so the exercised code paths
match the real tests. Only standard library + numpy + torch are imported; no
dependency on the parent superrepo (``aios``).
"""
import numpy as np
import torch as T

from gsp_rl.src.actors.actor import Actor

INPUT_SIZE = 8
OUTPUT_SIZE_DISCRETE = 4
OUTPUT_SIZE_CONTINUOUS = 2

CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 8,
    "MEM_SIZE": 100,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}


def make_actor(network: str) -> Actor:
    """Construct an Actor for the given algorithm name."""
    continuous = network in {"DDPG", "TD3"}
    output_size = OUTPUT_SIZE_CONTINUOUS if continuous else OUTPUT_SIZE_DISCRETE
    min_max_action = 1.0 if continuous else 1
    return Actor(
        id=1,
        config=CONFIG,
        network=network,
        input_size=INPUT_SIZE,
        output_size=output_size,
        min_max_action=min_max_action,
        meta_param_size=1,
    )


def fill_buffer(actor: Actor, n: int = 20, continuous: bool = False) -> None:
    """Store n random transitions in the actor's replay buffer."""
    output_size = OUTPUT_SIZE_CONTINUOUS if continuous else OUTPUT_SIZE_DISCRETE
    for _ in range(n):
        s = np.random.randn(INPUT_SIZE).astype(np.float32)
        a = (
            np.random.randn(output_size).astype(np.float32)
            if continuous
            else np.random.randint(0, output_size)
        )
        r = float(np.random.randn())
        s_ = np.random.randn(INPUT_SIZE).astype(np.float32)
        d = bool(np.random.rand() > 0.8)
        actor.store_transition(s, a, r, s_, d, actor.networks)


def get_params_snapshot(network: T.nn.Module) -> T.Tensor:
    """Return a flat tensor of all parameter values."""
    return T.cat([p.data.flatten().clone() for p in network.parameters()])


def one_learn_step(network: str) -> dict:
    """Run one fully-deterministic learn step and return loss + post-step params.

    Determinism contract: both buffer construction and the learn call are pinned
    with explicit numpy + torch seeds so the result is bit-reproducible on CPU/MPS.
    """
    # Pin construction + buffer fill.
    np.random.seed(0)
    T.manual_seed(0)
    actor = make_actor(network)
    fill_buffer(actor, n=64, continuous=False)

    # Re-pin immediately before the learn call so replay sampling + any torch RNG
    # draws are independent of how many randoms the buffer fill consumed.
    np.random.seed(123)
    T.manual_seed(123)
    if network == "DQN":
        loss = actor.learn_DQN(actor.networks)
    else:
        loss = actor.learn_DDQN(actor.networks)

    params = get_params_snapshot(actor.networks["q_eval"]).detach().cpu().numpy()
    return {"loss": float(loss), "params": params}
