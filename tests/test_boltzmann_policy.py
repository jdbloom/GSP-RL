"""Boltzmann soft-policy: softmax(Q/tau) action sampling (discrete SAC-analog)."""
import types
import numpy as np
from gsp_rl.src.actors.learning_aids import NetworkAids


def _agent(tau, action_space, q_vec):
    """Minimal object bound to the real boltzmann_action, with a stubbed Q-forward."""
    a = types.SimpleNamespace()
    a.boltzmann_temperature = tau
    a.action_space = action_space
    # stub networks['q_eval'].forward(state) -> tensor-like with .detach().cpu().numpy()
    import torch as T
    net = types.SimpleNamespace(device="cpu",
                                forward=lambda s: T.tensor(q_vec, dtype=T.float))
    a._networks = {"q_eval": net}
    a.boltzmann_action = NetworkAids.boltzmann_action.__get__(a)
    return a


def test_low_tau_concentrates_on_argmax():
    q = [0.0, 5.0, 0.0]  # argmax = index 1
    a = _agent(0.05, [0, 1, 2], q)
    draws = [a.boltzmann_action(np.zeros(3), a._networks) for _ in range(300)]
    assert draws.count(1) > 285, f"low tau should nearly always pick argmax, got {draws.count(1)}"


def test_high_tau_approaches_uniform():
    q = [0.0, 1.0, 0.0]
    a = _agent(100.0, [0, 1, 2], q)
    draws = [a.boltzmann_action(np.zeros(3), a._networks) for _ in range(600)]
    # each ~200; assert no action is starved and argmax isn't dominant
    counts = [draws.count(i) for i in range(3)]
    assert all(c > 120 for c in counts), f"high tau should be near-uniform, got {counts}"


def test_numerically_stable_large_q():
    q = [1000.0, 1001.0, 999.0]  # would overflow exp without max-subtraction
    a = _agent(1.0, [0, 1, 2], q)
    d = a.boltzmann_action(np.zeros(3), a._networks)
    assert d in (0, 1, 2)  # no NaN/overflow


def test_default_temperature_zero_is_greedy_path():
    # boltzmann is only invoked when tau>0; the default 0.0 uses the argmax path.
    # Assert the Hyperparameters default is 0.0 (bit-exact greedy).
    from gsp_rl.src.actors.learning_aids import Hyperparameters
    import inspect
    src = inspect.getsource(Hyperparameters.__init__)
    assert "boltzmann_temperature" in src and "0.0 if _bt is None" in src
