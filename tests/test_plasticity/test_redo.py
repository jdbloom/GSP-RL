"""ReDo (Sokar 2023) dormant-unit reset — recycles dead units to restore
plasticity. Resets incoming weights (re-init), zeros outgoing weights, and
clears Adam state for the reset units."""
import torch as T
import torch.nn as nn

from gsp_rl.src.actors.plasticity import redo_reset


class _Net(nn.Module):
    """Minimal fc1->relu->fc2->relu->fc3 net mirroring DDQN structure."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(5, 2)
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        return self.fc3(x)


def _force_dead_unit(net, layer, unit):
    """Make a unit always-dead: large negative bias so post-ReLU is 0."""
    with T.no_grad():
        getattr(net, layer).bias[unit] = -1e6
        getattr(net, layer).weight[unit].zero_()


def test_redo_resets_dead_unit_incoming_and_outgoing():
    net = _Net()
    _force_dead_unit(net, "fc1", 2)        # unit 2 of fc1 is dead
    # ensure no fc2 unit is dormant so its row-reinit can't touch fc2 col 2
    with T.no_grad():
        net.fc2.weight.abs_(); net.fc2.bias.fill_(0.1)
    batch = T.randn(32, 4)
    before_out_col = net.fc2.weight[:, 2].clone()
    # isolate the fc1 reset: a single layer-pair so the outgoing-zero is clean
    n = redo_reset(net, batch, [("fc1", "fc2")], tau=0.1)
    assert n >= 1                          # at least the forced-dead unit reset
    # incoming weights re-initialized (no longer all-zero / huge-neg bias)
    assert net.fc1.bias[2].item() > -1e5
    assert net.fc1.weight[2].abs().sum().item() > 0
    # outgoing weights zeroed (fc1 unit 2 -> fc2 column 2)
    assert net.fc2.weight[:, 2].abs().sum().item() == 0.0
    assert not T.equal(net.fc2.weight[:, 2], before_out_col)


def test_redo_noop_when_all_units_active():
    net = _Net()
    # drive with positive inputs + positive-ish weights so units fire
    with T.no_grad():
        net.fc1.weight.abs_(); net.fc1.bias.fill_(0.1)
        net.fc2.weight.abs_(); net.fc2.bias.fill_(0.1)
    batch = T.abs(T.randn(32, 4)) + 0.5
    n = redo_reset(net, batch, [("fc1", "fc2"), ("fc2", "fc3")], tau=0.1)
    assert n == 0


def test_redo_clears_adam_state_for_reset_unit():
    net = _Net()
    # build up some Adam state
    for _ in range(3):
        net(T.randn(8, 4)).sum().backward(); net.optimizer.step(); net.optimizer.zero_grad()
    _force_dead_unit(net, "fc1", 1)
    redo_reset(net, T.randn(32, 4), [("fc1", "fc2"), ("fc2", "fc3")], tau=0.1)
    st = net.optimizer.state.get(net.fc1.weight, {})
    if "exp_avg" in st:                    # reset row's Adam moments zeroed
        assert st["exp_avg"][1].abs().sum().item() == 0.0
