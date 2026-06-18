"""Critic-divergence stabilizer flags: CRITIC_LOSS, GRAD_CLIP_NORM,
REWARD_SCALE, Q_TARGET_CLIP. Defaults must reproduce exact legacy behavior
(MSE loss, no clipping, unit reward scale) so existing batches are unaffected.
"""
import os
import copy
import yaml
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
