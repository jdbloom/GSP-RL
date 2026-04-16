"""Tests for ACTOR_USE_LAYER_NORM flag — separate from GSP head LN.

H-13 closure: the existing GSP_USE_LAYER_NORM only controls LN in the GSP head's
DDPGActorNetwork. To test whether LayerNorm in the *main DDQN actor* trunk
stabilizes the cyclical-reward plasticity loss (independent of GSP head behavior),
we need a separate flag — ACTOR_USE_LAYER_NORM — that wires LN into the DDQN
(and DQN) network's trunk. This test file enforces that the two flags are
independent and that the actor honors the new flag.

Architecture target (mirrors DDPGActorNetwork.use_layer_norm placement):
    Linear(input_size, fc1) -> [LayerNorm(fc1)] -> ReLU
                            -> Linear(fc1, fc2) -> [LayerNorm(fc2)] -> ReLU
                            -> Linear(fc2, output_size)
"""
import copy
import os

import torch.nn as nn
import yaml

from gsp_rl.src.actors import Actor

containing_folder = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(containing_folder, "config.yml")

with open(config_path, "r") as file:
    BASE_CONFIG = yaml.safe_load(file)


def _ddqn_actor_args(config):
    return {
        "id": 1,
        "network": "DDQN",
        "input_size": 32,
        "output_size": 2,
        "meta_param_size": 2,
        "gsp": True,
        "recurrent_gsp": False,
        "attention": False,
        "gsp_input_size": 6,
        "gsp_output_size": 1,
        "gsp_look_back": 2,
        "gsp_sequence_length": 5,
        "config": config,
        "min_max_action": 1.0,
    }


def _has_layer_norm(net) -> bool:
    return any(isinstance(m, nn.LayerNorm) for m in net.modules())


def test_ddqn_actor_no_layer_norm_by_default():
    """With no ACTOR_USE_LAYER_NORM in config, the DDQN q_eval/q_next have no LayerNorm."""
    config = copy.deepcopy(BASE_CONFIG)
    actor = Actor(**_ddqn_actor_args(config))
    assert not _has_layer_norm(actor.networks["q_eval"]), \
        "DDQN q_eval should not have LayerNorm by default (legacy behavior)"
    assert not _has_layer_norm(actor.networks["q_next"]), \
        "DDQN q_next should not have LayerNorm by default (legacy behavior)"


def test_ddqn_actor_layer_norm_when_flag_true():
    """With ACTOR_USE_LAYER_NORM=True, DDQN q_eval and q_next both have LayerNorm modules."""
    config = copy.deepcopy(BASE_CONFIG)
    config["ACTOR_USE_LAYER_NORM"] = True
    actor = Actor(**_ddqn_actor_args(config))
    assert _has_layer_norm(actor.networks["q_eval"]), \
        "DDQN q_eval must have LayerNorm when ACTOR_USE_LAYER_NORM=True"
    assert _has_layer_norm(actor.networks["q_next"]), \
        "DDQN q_next must have LayerNorm when ACTOR_USE_LAYER_NORM=True"


def test_ddqn_actor_layer_norm_explicit_false():
    """Explicit ACTOR_USE_LAYER_NORM=False keeps legacy no-LN behavior."""
    config = copy.deepcopy(BASE_CONFIG)
    config["ACTOR_USE_LAYER_NORM"] = False
    actor = Actor(**_ddqn_actor_args(config))
    assert not _has_layer_norm(actor.networks["q_eval"])
    assert not _has_layer_norm(actor.networks["q_next"])


def test_actor_ln_independent_of_gsp_head_ln():
    """ACTOR_USE_LAYER_NORM and GSP_USE_LAYER_NORM are independent flags.

    Tests all 4 combinations of (actor_ln, gsp_head_ln) — verifies each flag only
    affects its own network. The GSP head is the DDPGActorNetwork stored under
    actor.gsp_networks (built in Actor.__init__ when gsp=True).
    """
    cases = [
        # (actor_ln, gsp_head_ln, actor_should_have_ln, gsp_should_have_ln)
        (False, False, False, False),
        (True,  False, True,  False),
        (False, True,  False, True),
        (True,  True,  True,  True),
    ]
    for actor_ln, gsp_ln, want_actor, want_gsp in cases:
        config = copy.deepcopy(BASE_CONFIG)
        config["ACTOR_USE_LAYER_NORM"] = actor_ln
        config["GSP_USE_LAYER_NORM"] = gsp_ln
        actor = Actor(**_ddqn_actor_args(config))
        actor_has = _has_layer_norm(actor.networks["q_eval"])
        # The GSP head networks are stored in actor.gsp_networks dict.
        # The DDPG-style head is keyed under 'actor' (per make_DDPG_networks).
        assert actor.gsp_networks is not None, "GSP head should be built when gsp=True"
        gsp_head_net = actor.gsp_networks.get("actor")
        assert gsp_head_net is not None, "Expected 'actor' key in gsp_networks dict"
        gsp_has = _has_layer_norm(gsp_head_net)
        assert actor_has == want_actor, (
            f"actor_ln={actor_ln} gsp_ln={gsp_ln}: "
            f"actor LN got {actor_has}, want {want_actor}"
        )
        assert gsp_has == want_gsp, (
            f"actor_ln={actor_ln} gsp_ln={gsp_ln}: "
            f"GSP head LN got {gsp_has}, want {want_gsp}"
        )


def test_dqn_actor_layer_norm_when_flag_true():
    """ACTOR_USE_LAYER_NORM also applies to DQN networks (same architecture as DDQN)."""
    config = copy.deepcopy(BASE_CONFIG)
    config["ACTOR_USE_LAYER_NORM"] = True
    args = _ddqn_actor_args(config)
    args["network"] = "DQN"
    actor = Actor(**args)
    assert _has_layer_norm(actor.networks["q_eval"]), \
        "DQN q_eval must have LayerNorm when ACTOR_USE_LAYER_NORM=True"
    assert _has_layer_norm(actor.networks["q_next"]), \
        "DQN q_next must have LayerNorm when ACTOR_USE_LAYER_NORM=True"
