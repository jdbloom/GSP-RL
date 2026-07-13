"""#53 Sub-project B — batched-agent forward: GSP-head + acting-path contracts.

These tests pin the equivalence guarantees the BATCHED_ACTOR_FORWARD call
sites (RL-CollectiveTransport Main.py acting loop + Agent.choose_agent_gsp)
rely on when they route R per-robot forwards through one stacked
choose_actions_batch call:

1. DDQN acting: batched greedy argmax == sequential greedy argmax.
2. GSP head (stateless DDPG scheme): batched deterministic predictions match
   the sequential choose_action outputs within fp tolerance (the batched
   matmul changes float-reduction order — the documented, expected drift —
   so equality is allclose(atol=1e-6), never bit-exact by contract).
3. Shape correctness across GSP output widths K (scalar delta_theta-style
   K=1, kinematics-style K=3, trajectory-style K=5): one (K,) row per robot.
4. The exploring epsilon branch draws one gate per robot, explorer choice
   draws interleaved in robot order — the #91 v2 contract: the flag-on
   np.random stream is IDENTICAL to the sequential loop.

Complements tests/test_actor/test_batch_choose_action.py (April 2026, main
action nets only) with the GSP-head coverage the #53-B call sites need.
"""

import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


def _config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }


def _make_gsp_actor(k, gsp_input_size=6, seed=0, noise=0.0, min_max_action=0.1):
    """DDQN actor + stateless DDPG-scheme GSP head of output width K.

    Mirrors the production GSP-N shape: the actor's augmented input grows by
    K and the head is built by build_gsp_network('DDPG').
    """
    T.manual_seed(seed)
    config = _config()
    config["NOISE"] = noise
    actor = Actor(
        id=0, config=config, network="DDQN",
        input_size=31, output_size=9,
        min_max_action=min_max_action, meta_param_size=1,
        gsp=True, recurrent_gsp=False,
        gsp_input_size=gsp_input_size, gsp_output_size=k,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )
    return actor


class TestDDQNActingBatchContract:
    """Batched DDQN acting on the GSP-augmented input width (31 + K)."""

    def test_greedy_argmax_matches_sequential(self):
        actor = _make_gsp_actor(k=1)
        rng = np.random.default_rng(42)
        obs = [rng.standard_normal(32).astype(np.float32) for _ in range(4)]

        sequential = [
            actor.choose_action(o, actor.networks, test=True) for o in obs
        ]
        batched = actor.choose_actions_batch(obs, actor.networks, test=True)

        assert batched == sequential
        assert all(isinstance(a, int) for a in batched)

    def test_exploring_branch_per_robot_gate_draws(self):
        """epsilon=1.0: every robot explores off its OWN gate draw, with the
        explorer np.random.choice draws interleaved per robot in the exact
        sequential order (#91 v2 RNG contract)."""
        actor = _make_gsp_actor(k=1)
        actor.epsilon = 1.0
        rng = np.random.default_rng(7)
        obs = [rng.standard_normal(32).astype(np.float32) for _ in range(4)]

        np.random.seed(123)
        batched = actor.choose_actions_batch(obs, actor.networks, test=False)

        # Sequential stream: gate_i then choice_i, per robot in order.
        np.random.seed(123)
        expected = []
        for _ in range(4):
            _ = np.random.random()
            expected.append(np.random.choice(actor.action_space))

        assert len(batched) == 4
        assert all(a in actor.action_space for a in batched)
        assert batched == expected


class TestGSPHeadBatchEquivalence:
    """Batched GSP-head prediction vs the sequential per-agent loop."""

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_deterministic_predictions_match_sequential(self, k):
        actor = _make_gsp_actor(k=k)
        rng = np.random.default_rng(k)
        states = [rng.standard_normal(6).astype(np.float32) for _ in range(4)]

        sequential = np.array([
            actor.choose_action(s, actor.gsp_networks, test=True)
            for s in states
        ])
        batched = actor.choose_actions_batch(
            states, actor.gsp_networks, test=True)

        # Documented expected drift: batched matmul changes float-reduction
        # order — fp-tolerance equality, not bit-exact.
        np.testing.assert_allclose(batched, sequential, atol=1e-6)

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_shape_one_k_row_per_robot(self, k):
        actor = _make_gsp_actor(k=k)
        rng = np.random.default_rng(k + 100)
        states = [rng.standard_normal(6).astype(np.float32) for _ in range(4)]

        batched = actor.choose_actions_batch(
            states, actor.gsp_networks, test=True)

        assert batched.shape == (4, k)
        # The call site consumes list(batched) — one (K,) vector per robot,
        # same as the sequential choose_action return.
        rows = list(batched)
        assert len(rows) == 4
        assert all(r.shape == (k,) for r in rows)

    def test_clamp_matches_sequential_bound(self):
        """Both paths clamp gsp predictions to ±min_max_action (the legacy
        sequential quirk: gsp_networks predictions share the ACTION bound)."""
        actor = _make_gsp_actor(k=1, min_max_action=0.1)
        rng = np.random.default_rng(3)
        # Large inputs push tanh-bounded outputs toward ±1 → clamp engages.
        states = [
            (10.0 * rng.standard_normal(6)).astype(np.float32)
            for _ in range(4)
        ]

        sequential = np.array([
            actor.choose_action(s, actor.gsp_networks, test=True)
            for s in states
        ])
        batched = actor.choose_actions_batch(
            states, actor.gsp_networks, test=True)

        assert np.all(np.abs(batched) <= 0.1 + 1e-7)
        np.testing.assert_allclose(batched, sequential, atol=1e-6)

    def test_noise_path_stays_within_clamp(self):
        """Training-mode (test=False) batched predictions are still bounded;
        the noise stream itself is a documented RNG-contract change."""
        actor = _make_gsp_actor(k=1, noise=0.1)
        rng = np.random.default_rng(5)
        states = [rng.standard_normal(6).astype(np.float32) for _ in range(4)]

        T.manual_seed(0)
        batched = actor.choose_actions_batch(
            states, actor.gsp_networks, test=False)

        assert batched.shape == (4, 1)
        assert np.all(np.abs(batched) <= actor.min_max_action + 1e-7)

    def test_attention_gsp_head_refuses_batching(self):
        """Attention (A-GSP) heads must raise — the call-site gate relies on
        this being loud, never a silent wrong answer."""
        T.manual_seed(0)
        actor = Actor(
            id=0, config=_config(), network="DDQN",
            input_size=31, output_size=9,
            min_max_action=0.1, meta_param_size=1,
            gsp=True, recurrent_gsp=False, attention=True,
            gsp_input_size=6, gsp_output_size=1,
            gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        )
        states = [np.zeros(6, dtype=np.float32) for _ in range(4)]
        with pytest.raises(NotImplementedError):
            actor.choose_actions_batch(states, actor.gsp_networks, test=True)
