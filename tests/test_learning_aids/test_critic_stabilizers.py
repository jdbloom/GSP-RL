"""Critic-divergence stabilizer flags: CRITIC_LOSS, GRAD_CLIP_NORM,
REWARD_SCALE, Q_TARGET_CLIP. Defaults must reproduce exact legacy behavior
(MSE loss, no clipping, unit reward scale) so existing batches are unaffected.
"""
import os
import copy
import yaml
import torch as T
import torch.nn as nn

from gsp_rl.src.networks import DDQN
from gsp_rl.src.actors import NetworkAids

_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml')
with open(_cfg_path, 'r') as _f:
    BASE_CONFIG = yaml.safe_load(_f)


# --- network-level loss switch ---------------------------------------------

def test_ddqn_default_loss_is_mse():
    net = DDQN(id=1, lr=1e-4, input_size=8, output_size=4)
    assert isinstance(net.loss, nn.MSELoss)


def test_ddqn_huber_loss_when_requested():
    net = DDQN(id=1, lr=1e-4, input_size=8, output_size=4, critic_loss='huber')
    assert isinstance(net.loss, nn.SmoothL1Loss)


def test_ddqn_unknown_critic_loss_falls_back_to_mse():
    net = DDQN(id=1, lr=1e-4, input_size=8, output_size=4, critic_loss='bogus')
    assert isinstance(net.loss, nn.MSELoss)


# --- Hyperparameters defaults (legacy no-ops) ------------------------------

def test_stabilizer_defaults_are_noops():
    na = NetworkAids(copy.deepcopy(BASE_CONFIG))
    assert na.critic_loss == 'mse'
    assert na.grad_clip_norm == 0.0
    assert na.reward_scale == 1.0
    assert na.q_target_clip == 0.0


def test_stabilizer_overrides_are_read():
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update({'CRITIC_LOSS': 'Huber', 'GRAD_CLIP_NORM': 1.0,
                'REWARD_SCALE': 0.01, 'Q_TARGET_CLIP': 500.0})
    na = NetworkAids(cfg)
    assert na.critic_loss == 'huber'        # lowercased
    assert na.grad_clip_norm == 1.0
    assert na.reward_scale == 0.01
    assert na.q_target_clip == 500.0


# --- nn_args wiring: make_DDQN_networks honors critic_loss -----------------

def test_make_ddqn_networks_threads_huber():
    na = NetworkAids(copy.deepcopy(BASE_CONFIG))
    nets = na.make_DDQN_networks(
        {'id': 1, 'lr': 1e-4, 'input_size': 8, 'output_size': 4,
         'critic_loss': 'huber'})
    assert isinstance(nets['q_eval'].loss, nn.SmoothL1Loss)
    assert isinstance(nets['q_next'].loss, nn.SmoothL1Loss)


# --- shared critic helpers (used by ALL of DQN/DDQN/DDPG/RDDPG/TD3) ---------

def test_shared_critic_loss_fn_switches_with_flag():
    assert isinstance(NetworkAids(copy.deepcopy(BASE_CONFIG))._critic_loss_fn, nn.MSELoss)
    cfg = copy.deepcopy(BASE_CONFIG); cfg['CRITIC_LOSS'] = 'huber'
    assert isinstance(NetworkAids(cfg)._critic_loss_fn, nn.SmoothL1Loss)


def test_q_target_default_is_plain_bellman():
    na = NetworkAids(copy.deepcopy(BASE_CONFIG))  # reward_scale=1, clip=0
    r, boot = T.tensor([1.0, -2.0]), T.tensor([10.0, 20.0])
    assert T.allclose(na._q_target(r, boot), r + na.gamma * boot)


def test_q_target_applies_reward_scale_then_clip():
    cfg = copy.deepcopy(BASE_CONFIG); cfg.update({'REWARD_SCALE': 0.5, 'Q_TARGET_CLIP': 5.0})
    na = NetworkAids(cfg)
    # 0.5*100 + gamma*0 = 50, clamped to 5
    assert T.allclose(na._q_target(T.tensor([100.0]), T.tensor([0.0])), T.tensor([5.0]))


def test_clip_critic_grad_noop_when_disabled():
    na = NetworkAids(copy.deepcopy(BASE_CONFIG))  # grad_clip_norm=0
    net = DDQN(id=1, lr=1e-4, input_size=4, output_size=2)
    net((T.ones(1, 4) * 50).to(net.device)).sum().backward()
    g = net.fc1.weight.grad.norm().item()
    na._clip_critic_grad(net)
    assert net.fc1.weight.grad.norm().item() == g


def test_clip_critic_grad_clips_when_enabled():
    cfg = copy.deepcopy(BASE_CONFIG); cfg['GRAD_CLIP_NORM'] = 0.001
    na = NetworkAids(cfg)
    net = DDQN(id=1, lr=1e-4, input_size=4, output_size=2)
    net((T.ones(1, 4) * 50).to(net.device)).sum().backward()
    na._clip_critic_grad(net)
    assert net.fc1.weight.grad.norm().item() <= 0.001 + 1e-6


# --- numeric hyperparameter coercion (YAML sci-notation -> str crash class) ---

def test_string_lr_coerced_in_hyperparameters():
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update({'LR': '3e-05', 'ALPHA': '1e-3', 'BETA': '2e-3',
                'GAMMA': '0.99', 'TAU': '5e-3'})
    na = NetworkAids(cfg)
    for v in (na.lr, na.alpha, na.beta, na.gamma, na.tau):
        assert isinstance(v, float)
    assert na.lr == 3e-05


def test_ddqn_constructs_with_string_lr():
    # '3e-05' as a str previously crashed Adam('<=' float vs str)
    net = DDQN(id=1, lr='3e-05', input_size=4, output_size=2)
    assert net.optimizer.param_groups[0]['lr'] == 3e-05


# --- diagnostics now default ON (instrument from day one) -------------------

def test_diagnostics_enabled_defaults_true():
    na = NetworkAids(copy.deepcopy(BASE_CONFIG))
    assert na.diagnostics_enabled is True


def test_diagnostics_can_be_disabled():
    cfg = copy.deepcopy(BASE_CONFIG); cfg['DIAGNOSTICS_ENABLED'] = False
    assert NetworkAids(cfg).diagnostics_enabled is False
