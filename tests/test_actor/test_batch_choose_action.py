"""Tests for batched action selection — verify it produces same results as sequential."""

import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


@pytest.fixture
def config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }


class TestDQNBatch:
    def test_batch_matches_sequential(self, config):
        """Batched DQN should return identical actions to sequential calls."""
        config["EPSILON"] = 0.0  # greedy — no randomness
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = [actor.choose_action(obs, actor.networks, test=True) for obs in observations]
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        assert sequential == batched

    def test_batch_returns_correct_count(self, config):
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(6)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert len(actions) == 6

    def test_batch_actions_in_range(self, config):
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=9, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        for a in actions:
            assert 0 <= a < 9


class TestDDQNBatch:
    def test_batch_matches_sequential(self, config):
        config["EPSILON"] = 0.0
        actor = Actor(id=1, config=config, network="DDQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = [actor.choose_action(obs, actor.networks, test=True) for obs in observations]
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        assert sequential == batched

    def test_all_exploit_step_consumes_per_robot_gate_draws_and_goes_greedy(self, config):
        """0 < epsilon < 1, a seed whose FOUR gate draws all EXPLOIT: exactly
        R np.random.random() gate draws (one per robot — the #91 v2 contract),
        ZERO np.random.choice draws, then the batched greedy actions."""
        config["EPSILON"] = 0.5
        actor = Actor(id=1, config=config, network="DDQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        rng = np.random.default_rng(7)
        observations = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]

        # Greedy reference: the test=True path never touches np.random.
        greedy = actor.choose_actions_batch(observations, actor.networks, test=True)

        np.random.seed(0)  # first four np.random.random() draws all > 0.5
        batched = actor.choose_actions_batch(observations, actor.networks, test=False)
        state_after = np.random.get_state()

        np.random.seed(0)
        gates = [np.random.random() for _ in range(4)]
        assert all(g > actor.epsilon for g in gates)  # seed sanity: all exploit
        expected_state = np.random.get_state()

        assert batched == greedy
        assert state_after[0] == expected_state[0]
        np.testing.assert_array_equal(state_after[1], expected_state[1])
        assert state_after[2:] == expected_state[2:]

    def test_mixed_step_np_random_stream_identical_to_sequential(self, config):
        """#91 v2 acceptance gate: on a MIXED explore/exploit step, flag-on
        actions AND the post-call np.random state both equal the sequential
        choose_action loop's — the RNG stream is IDENTICAL, not just the same
        draw count."""
        config["EPSILON"] = 0.6
        actor = Actor(id=1, config=config, network="DDQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        rng = np.random.default_rng(11)
        observations = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]

        np.random.seed(0)
        batched = actor.choose_actions_batch(observations, actor.networks, test=False)
        state_batched = np.random.get_state()

        np.random.seed(0)
        sequential = [actor.choose_action(obs, actor.networks, test=False)
                      for obs in observations]
        state_sequential = np.random.get_state()

        # Seed sanity: this seed/epsilon mixes both branches across the step.
        np.random.seed(0)
        exploits = []
        for _ in range(4):
            exploit = np.random.random() > actor.epsilon
            exploits.append(exploit)
            if not exploit:
                np.random.choice(actor.action_space)
        assert any(exploits) and not all(exploits)

        assert batched == sequential
        assert state_batched[0] == state_sequential[0]
        np.testing.assert_array_equal(state_batched[1], state_sequential[1])
        assert state_batched[2:] == state_sequential[2:]


class TestDDPGBatch:
    def test_batch_matches_sequential(self, config):
        config["NOISE"] = 0.0
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = np.array([actor.choose_action(obs, actor.networks, test=True) for obs in observations])
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        np.testing.assert_allclose(sequential, batched, atol=1e-5)

    def test_batch_output_shape(self, config):
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert actions.shape == (4, 2)


class TestTD3Batch:
    def test_batch_matches_sequential_after_warmup(self, config):
        config["NOISE"] = 0.0
        config["WARMUP"] = 0
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        # Advance past warmup
        actor.time_step = 100
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        # Sequential calls each increment time_step, so save/restore
        saved_ts = actor.time_step
        sequential = []
        for obs in observations:
            actor.time_step = saved_ts  # reset to same point
            sequential.append(actor.choose_action(obs, actor.networks, test=True))
        sequential = np.array(sequential)

        actor.time_step = saved_ts
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        np.testing.assert_allclose(sequential, batched, atol=1e-5)

    def test_batch_output_shape(self, config):
        config["WARMUP"] = 0
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        actor.time_step = 100
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert actions.shape == (4, 2)


class TestBatchNotSupported:
    def test_rddpg_raises(self, config):
        """RDDPG should explicitly refuse batching."""
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        # The main networks are DDPG (batchable), but gsp_networks are RDDPG
        with pytest.raises(NotImplementedError):
            actor.choose_actions_batch(observations, actor.gsp_networks, test=True)
