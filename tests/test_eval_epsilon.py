"""Test EVAL_EPSILON knob on choose_action for DQN/DDQN."""
import numpy as np
import pytest

from gsp_rl.src.actors.actor import Actor


class MinimalActor:
    """Stand-in holding just the attributes needed by choose_action."""
    def __init__(self, epsilon, eval_epsilon, action_space):
        self.epsilon = epsilon
        self.eval_epsilon = eval_epsilon
        self.action_space = action_space

    # DO NOT define choose_action here; bind the real method below.
    def DQN_DDQN_choose_action(self, observation, networks):
        return self._greedy_sentinel

# Bind the REAL choose_action from Actor so the test exercises the edited code.
MinimalActor.choose_action = Actor.choose_action


@pytest.mark.parametrize("n_draws", [200])
def test_eval_epsilon_zero_is_pure_greedy(n_draws):
    """eval_epsilon=0.0, test=True → always the greedy output."""
    agent = MinimalActor(epsilon=0.5, eval_epsilon=0.0, action_space=[0, 1, 2])
    sentinel = 7  # not in action_space
    agent._greedy_sentinel = sentinel
    fake_networks = {"learning_scheme": "DQN"}
    results = [agent.choose_action(np.zeros(4), fake_networks, test=True) for _ in range(n_draws)]
    assert all(a == sentinel for a in results), "eval_epsilon=0 must always call the greedy path"


def test_eval_epsilon_one_is_uniform_random():
    """eval_epsilon=1.0, test=True → actions are always drawn uniformly from action_space."""
    agent = MinimalActor(epsilon=0.5, eval_epsilon=1.0, action_space=[0, 1, 2])
    sentinel = 7
    agent._greedy_sentinel = sentinel
    fake_networks = {"learning_scheme": "DDQN"}
    results = [agent.choose_action(np.zeros(4), fake_networks, test=True) for _ in range(200)]
    assert sentinel not in results, "greedy path must never be taken"
    assert all(a in agent.action_space for a in results), "all actions must be from action_space"


def test_training_unchanged():
    """training path (test=False) with epsilon=0.0 is always greedy."""
    # Patch np.random.random to return 0.5 so the condition "random>0.0" is true.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.random, "random", lambda: 0.5)
        agent = MinimalActor(epsilon=0.0, eval_epsilon=0.5, action_space=[0, 1, 2])
        sentinel = 7
        agent._greedy_sentinel = sentinel
        fake_networks = {"learning_scheme": "DQN"}
        results = [agent.choose_action(np.zeros(4), fake_networks, test=False) for _ in range(200)]
        assert all(a == sentinel for a in results), "training path must use greedy when epsilon=0"