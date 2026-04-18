"""Tests for GSP_PREDICTION_TARGET and GSP_PREDICTION_HORIZON flags.

Candidate A architecture pivot: instead of training the GSP head to predict
collective Δθ (the self-referential target that collapses to constant-zero),
the head can be retargeted to predict the robot's own future proximity at
horizon K. The flag is read by Hyperparameters; the actual delayed-label
construction happens in the host code (RL-CollectiveTransport agent.py).

Default 'delta_theta' preserves legacy behavior for all existing experiments.
"""
import copy
import os

import yaml

from gsp_rl.src.actors.learning_aids import Hyperparameters

containing_folder = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(containing_folder, "config.yml")

with open(config_path, "r") as f:
    BASE_CONFIG = yaml.safe_load(f)


def test_prediction_target_default_delta_theta():
    config = copy.deepcopy(BASE_CONFIG)
    hp = Hyperparameters(config)
    assert hp.gsp_prediction_target == "delta_theta"


def test_prediction_horizon_default_five():
    config = copy.deepcopy(BASE_CONFIG)
    hp = Hyperparameters(config)
    assert hp.gsp_prediction_horizon == 5


def test_prediction_target_override_future_prox():
    config = copy.deepcopy(BASE_CONFIG)
    config["GSP_PREDICTION_TARGET"] = "future_prox"
    hp = Hyperparameters(config)
    assert hp.gsp_prediction_target == "future_prox"


def test_prediction_horizon_override():
    config = copy.deepcopy(BASE_CONFIG)
    config["GSP_PREDICTION_HORIZON"] = 10
    hp = Hyperparameters(config)
    assert hp.gsp_prediction_horizon == 10
