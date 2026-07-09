"""Tests for the K-step GLOBAL trajectory GSP output kinds
(goal_progress_traj, cyl_displacement_traj).

Motivation (2026-07-09, force-causal-use campaign): the payload-rotation
trajectory (delta_theta_traj) is learnable + coupled but value-null. The
objective variables of the task are GLOBAL: the payload's translation and its
progress-to-goal. These two kinds mirror the delta_theta_traj machinery
exactly — delayed-FIFO labels built by the host (RL-CollectiveTransport
Main.py), horizon-coupled head output width — but predict:

  goal_progress_traj     O = K   per-step payload progress-to-goal delta
                                 (prev_cyl_dist2goal - curr, positive = toward
                                 goal) over the NEXT K steps.
  cyl_displacement_traj  O = 2K  per-step payload (dx, dy) over the next K
                                 steps, flattened [dx1,dy1,...,dxK,dyK].

Labels are RAW physical units (meters) — no magic scaling; the loss-balance
lambda is set from measured label stds (F15 lesson).

K = GSP_PREDICTION_HORIZON. Everything that derives from gsp_network_output
(actor input width, replay gsp_label_size incl. the TD3 path, the
RunningStandardizer dim) must handle size K and 2K.
"""
import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.actor import Actor


BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.001,
    "LR": 1e-4,
    "EPSILON": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DEC": 1e-4,
    "BATCH_SIZE": 16,
    "MEM_SIZE": 200,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.1,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}

# (kind == target name, horizon multiplier)
_GLOBAL_TRAJ_CASES = [
    ("goal_progress_traj", 1),
    ("cyl_displacement_traj", 2),
]


def _make_actor_cfg(network="DDQN", **overrides) -> Actor:
    config = {**BASE_CONFIG, **overrides}
    return Actor(
        id=0, config=config, network=network, input_size=31, output_size=9,
        min_max_action=1, meta_param_size=32, gsp=True, gsp_input_size=6,
        gsp_output_size=1,
    )


# ---------------------------------------------------------------------------
# Horizon-coupled sizing: O == mult * K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,mult", _GLOBAL_TRAJ_CASES)
class TestGlobalTrajKindSizes:
    def test_output_size_equals_mult_times_horizon(self, kind, mult):
        for K in (1, 3, 5):
            actor = _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=K)
            assert actor.gsp_output_kind == kind
            assert actor.gsp_output_size_effective == mult * K, f"K={K}"
            assert actor.gsp_network_output == mult * K, f"K={K}"

    def test_head_mu_out_features(self, kind, mult):
        actor = _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=3)
        assert actor.gsp_networks["actor"].mu.out_features == mult * 3

    def test_actor_input_size_augmented(self, kind, mult):
        for K in (1, 3, 5):
            actor = _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=K)
            assert actor.network_input_size == 31 + mult * K, f"K={K}"

    def test_gsp_replay_label_slot_width(self, kind, mult):
        """The GSP head's own replay stores the label in the action slot; the
        slot must be (mult*K) wide (the dtraj crash site: `could not broadcast
        (K,) into (1,)`)."""
        actor = _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=4)
        buf = actor.gsp_networks["replay"]
        assert buf.action_memory.shape == (actor.mem_size, mult * 4)

    def test_horizon_zero_rejected(self, kind, mult):
        with pytest.raises(ValueError, match="GSP_PREDICTION_HORIZON"):
            _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=0)


# ---------------------------------------------------------------------------
# Target -> kind auto-derive + consistency (mirror of delta_theta_traj)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,mult", _GLOBAL_TRAJ_CASES)
class TestGlobalTrajTargetKindConsistency:
    def test_target_autoderives_kind_and_sizes_head(self, target, mult):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target)  # no GSP_OUTPUT_KIND
        assert actor.gsp_output_kind == target
        assert actor.gsp_network_output == mult * 5  # default horizon
        assert actor.gsp_networks["actor"].mu.out_features == mult * 5

    def test_target_respects_horizon(self, target, mult):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                                GSP_PREDICTION_HORIZON=3)
        assert actor.gsp_network_output == mult * 3

    def test_consistent_explicit_kind_ok(self, target, mult):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                                GSP_OUTPUT_KIND=target)
        assert actor.gsp_network_output == mult * 5

    def test_contradictory_kind_raises(self, target, mult):
        with pytest.raises(ValueError,
                           match=f"requires GSP_OUTPUT_KIND='{target}'"):
            _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                            GSP_OUTPUT_KIND="future_prox_1d")

    def test_cross_traj_kind_contradiction_raises(self, target, mult):
        """An explicit DIFFERENT trajectory kind is still a contradiction."""
        other = ("delta_theta_traj" if target != "delta_theta_traj"
                 else "goal_progress_traj")
        with pytest.raises(ValueError,
                           match=f"requires GSP_OUTPUT_KIND='{target}'"):
            _make_actor_cfg(GSP_PREDICTION_TARGET=target, GSP_OUTPUT_KIND=other)


# ---------------------------------------------------------------------------
# E2E replay gsp_label column width (DDQN + the previously-missing TD3 path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,mult", _GLOBAL_TRAJ_CASES)
class TestE2EReplayLabelWidth:
    def test_ddqn_e2e_main_replay_label_width(self, target, mult):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                                GSP_PREDICTION_HORIZON=5,
                                GSP_E2E_ENABLED=True)
        buf = actor.networks["replay"]
        assert buf.gsp_label_memory.shape == (actor.mem_size, mult * 5)
        assert buf.gsp_obs_memory.shape == (actor.mem_size, actor.gsp_network_input)

    def test_td3_e2e_main_replay_label_width(self, target, mult):
        """TD3 path previously omitted gsp_label_size (defaulted to 1); a K- or
        2K-wide E2E label must get a matching column."""
        actor = _make_actor_cfg(network="TD3",
                                GSP_PREDICTION_TARGET=target,
                                GSP_PREDICTION_HORIZON=5,
                                GSP_E2E_ENABLED=True)
        buf = actor.networks["replay"]
        assert buf.gsp_label_memory.shape == (actor.mem_size, mult * 5)


def test_td3_e2e_scalar_label_width_unchanged():
    """Legacy scalar target + TD3 e2e: label column stays width 1 (regression)."""
    actor = _make_actor_cfg(network="TD3", GSP_E2E_ENABLED=True)
    buf = actor.networks["replay"]
    assert buf.gsp_label_memory.shape == (actor.mem_size, 1)


# ---------------------------------------------------------------------------
# RunningStandardizer dim follows the slot width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,mult", _GLOBAL_TRAJ_CASES)
def test_feature_stats_dim_matches_slot(target, mult):
    actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                            GSP_PREDICTION_HORIZON=5,
                            GSP_E2E_ENABLED=True,
                            GSP_E2E_NORMALIZE_FEATURE=True)
    assert actor.gsp_feature_stats is not None
    assert actor.gsp_feature_stats.dim == mult * 5


# ---------------------------------------------------------------------------
# Head prediction width fills the actor's GSP augmentation slot (2K case)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,mult", _GLOBAL_TRAJ_CASES)
def test_head_prediction_fills_actor_gsp_slot(target, mult):
    for K in (1, 3, 5):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET=target,
                                GSP_PREDICTION_HORIZON=K)
        gsp_slot = actor.network_input_size - actor.input_size
        assert gsp_slot == mult * K, f"K={K}: slot {gsp_slot} != {mult * K}"
        gsp_net = actor.gsp_networks["actor"]
        obs = T.zeros(actor.gsp_network_input, dtype=T.float32).to(gsp_net.device)
        with T.no_grad():
            pred = gsp_net(obs)
        pred_w = int(np.asarray(pred.cpu()).ravel().shape[0])
        assert pred_w == gsp_slot, f"K={K}: pred width {pred_w} != slot {gsp_slot}"


# ---------------------------------------------------------------------------
# Regression: existing kinds byte-identical
# ---------------------------------------------------------------------------

def test_existing_kind_sizes_unchanged():
    for kind, expected in [
        ("delta_theta_1d", 1), ("future_prox_1d", 1), ("cyl_kinematics_3d", 3),
        ("cyl_kinematics_goal_4d", 4), ("time_to_goal_1d", 1),
        ("neighbor_force_1d", 1),
    ]:
        actor = _make_actor_cfg(GSP_OUTPUT_KIND=kind, GSP_PREDICTION_HORIZON=7)
        assert actor.gsp_network_output == expected, kind


def test_delta_theta_traj_still_K_not_2K():
    actor = _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj",
                            GSP_PREDICTION_HORIZON=4)
    assert actor.gsp_network_output == 4


def test_scalar_target_still_unaffected():
    actor = _make_actor_cfg(GSP_PREDICTION_TARGET="future_prox")
    assert actor.gsp_network_output == 1
