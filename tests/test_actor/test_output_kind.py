"""Tests for GSP_OUTPUT_KIND config flag and its effect on the GSP head's output_size.

Verifies He 2509.22335 Theorem 6.2 motivation: increasing output dimension O
raises the achievable Hessian rank ceiling, potentially breaking rank-1 collapse.

Each output kind must produce a GSP head whose DDPGActorNetwork output_size
(mu layer out_features) matches the documented target dimensionality.
"""
import pytest
from gsp_rl.src.actors.actor import Actor


# Minimal config — only required keys for Actor.__init__ + Hyperparameters.
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

# (gsp_output_kind, expected_output_size)
_OUTPUT_KIND_CASES = [
    ("delta_theta_1d", 1),
    ("future_prox_1d", 1),
    ("cyl_kinematics_3d", 3),
    ("cyl_kinematics_goal_4d", 4),
    ("time_to_goal_1d", 1),
    ("neighbor_force_1d", 1),
]


def _make_actor(gsp_output_kind: str) -> Actor:
    config = {**BASE_CONFIG, "GSP_OUTPUT_KIND": gsp_output_kind}
    return Actor(
        id=0,
        config=config,
        network="DDQN",
        input_size=31,
        output_size=9,
        min_max_action=1,
        meta_param_size=32,
        gsp=True,
        gsp_input_size=6,
        gsp_output_size=1,  # kwarg is the legacy default; kind overrides for non-1d
    )


@pytest.mark.parametrize("kind,expected_out", _OUTPUT_KIND_CASES)
class TestOutputKindHeadSize:
    """GSP head output_size matches expected value for each kind."""

    def test_gsp_network_output_attr(self, kind, expected_out):
        actor = _make_actor(kind)
        assert actor.gsp_network_output == expected_out, (
            f"gsp_output_kind='{kind}': expected gsp_network_output={expected_out}, "
            f"got {actor.gsp_network_output}"
        )

    def test_ddpg_actor_mu_layer_out_features(self, kind, expected_out):
        """The DDPGActorNetwork mu layer's out_features must match expected_out."""
        actor = _make_actor(kind)
        assert actor.gsp_networks is not None, "gsp_networks must be built when gsp=True"
        gsp_actor_net = actor.gsp_networks["actor"]
        # DDPGActorNetwork.mu is the final linear projection
        mu_out = gsp_actor_net.mu.out_features
        assert mu_out == expected_out, (
            f"gsp_output_kind='{kind}': DDPGActorNetwork mu.out_features={mu_out}, "
            f"expected {expected_out}"
        )

    def test_actor_input_size_augmented(self, kind, expected_out):
        """Actor's augmented input size = base_input_size + gsp_output_size_effective."""
        actor = _make_actor(kind)
        # input_size=31, gsp adds expected_out dims
        assert actor.network_input_size == 31 + expected_out, (
            f"gsp_output_kind='{kind}': network_input_size={actor.network_input_size}, "
            f"expected {31 + expected_out}"
        )


class TestBackwardCompatDefaultKind:
    """Default GSP_OUTPUT_KIND='delta_theta_1d' is identical to no-GSP_OUTPUT_KIND runs."""

    def test_default_kind_equals_no_kind(self):
        """Actor built without GSP_OUTPUT_KIND must match one with explicit default."""
        actor_no_kind = Actor(
            id=0,
            config=BASE_CONFIG,
            network="DDQN",
            input_size=31,
            output_size=9,
            min_max_action=1,
            meta_param_size=32,
            gsp=True,
            gsp_input_size=6,
            gsp_output_size=1,
        )
        actor_explicit = _make_actor("delta_theta_1d")
        assert actor_no_kind.gsp_network_output == actor_explicit.gsp_network_output == 1
        assert actor_no_kind.network_input_size == actor_explicit.network_input_size == 32

    def test_default_kind_mu_out_features(self):
        actor = _make_actor("delta_theta_1d")
        assert actor.gsp_networks["actor"].mu.out_features == 1

    def test_default_gsp_output_size_effective(self):
        """Hyperparameters sets gsp_output_size_effective=1 for default kind."""
        actor = _make_actor("delta_theta_1d")
        assert actor.gsp_output_size_effective == 1


class TestInvalidKindRaises:
    def test_unknown_kind_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown GSP_OUTPUT_KIND"):
            _make_actor("not_a_valid_kind")


def _make_actor_cfg(**overrides) -> Actor:
    """Build a GSP Actor with arbitrary config overrides."""
    config = {**BASE_CONFIG, **overrides}
    return Actor(
        id=0, config=config, network="DDQN", input_size=31, output_size=9,
        min_max_action=1, meta_param_size=32, gsp=True, gsp_input_size=6,
        gsp_output_size=1,
    )


class TestPredictionTargetKindConsistency:
    """GSP_PREDICTION_TARGET='delta_theta_traj' must yield a size-K head + buffer.

    Regression for 2026-07-08: the dtraj arm set GSP_PREDICTION_TARGET=delta_theta_traj
    (host emits a size-K label) but left GSP_OUTPUT_KIND at the scalar default, so the
    GSP replay buffer's action/label slot was width-1 and training crashed mid-episode
    with `could not broadcast (K,) into (1,)`. The output dim is fully determined by the
    target, so the kind is now auto-derived (and an explicit contradiction rejected).
    """

    def test_traj_target_autoderives_kind_and_sizes_head(self):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj")  # no GSP_OUTPUT_KIND
        assert actor.gsp_output_kind == "delta_theta_traj"
        assert actor.gsp_network_output == 5  # default GSP_PREDICTION_HORIZON
        assert actor.gsp_networks["actor"].mu.out_features == 5

    def test_traj_target_buffer_label_slot_is_size_k(self):
        """The GSP replay buffer's action/label slot must be width-K (the crash site)."""
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj",
                                 GSP_PREDICTION_HORIZON=5)
        buf = actor.gsp_networks["replay"]
        assert buf.action_memory.shape == (actor.mem_size, 5), buf.action_memory.shape

    def test_traj_target_respects_horizon(self):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj",
                                 GSP_PREDICTION_HORIZON=3)
        assert actor.gsp_network_output == 3
        assert actor.gsp_networks["replay"].action_memory.shape[1] == 3

    def test_consistent_explicit_kind_ok(self):
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj",
                                 GSP_OUTPUT_KIND="delta_theta_traj")
        assert actor.gsp_network_output == 5

    def test_contradictory_kind_raises(self):
        with pytest.raises(ValueError, match="requires GSP_OUTPUT_KIND='delta_theta_traj'"):
            _make_actor_cfg(GSP_PREDICTION_TARGET="delta_theta_traj",
                            GSP_OUTPUT_KIND="future_prox_1d")

    def test_scalar_target_unaffected(self):
        """A scalar target (future_prox) is dimensionally safe and left untouched."""
        actor = _make_actor_cfg(GSP_PREDICTION_TARGET="future_prox")
        assert actor.gsp_network_output == 1


class TestActorStateWidthInvariant:
    """The runtime GSP head PREDICTION width must exactly fill the actor Q-net's
    GSP augmentation slot.

    Regression for the 2026-07-08 actor-forward crash (`mat1 and mat2 shapes
    cannot be multiplied (64x40 and 36x64)`): the host concatenates the head's
    size-K prediction onto the base obs to form the actor state, and the actor
    net was built with network_input_size = input_size + gsp_network_output.
    These must agree for K>1, i.e. the head's actual forward-pass output width
    equals (network_input_size - input_size). This asserts the GSP-RL half of
    the cross-repo invariant (the RL-CT host asserts make_agent_state matches).
    """

    def test_head_prediction_fills_actor_gsp_slot(self):
        import numpy as np
        import torch as T
        for K in (1, 3, 5):
            actor = _make_actor_cfg(
                GSP_PREDICTION_TARGET="delta_theta_traj",
                GSP_PREDICTION_HORIZON=K,
            )
            gsp_slot = actor.network_input_size - actor.input_size
            assert gsp_slot == K, (
                f"K={K}: actor GSP augmentation slot {gsp_slot} != K"
            )
            # Run the actual GSP head forward; its prediction width must equal
            # the slot the actor net reserved for it.
            gsp_net = actor.gsp_networks["actor"]
            obs = T.zeros(actor.gsp_network_input, dtype=T.float32).to(gsp_net.device)
            with T.no_grad():
                pred = gsp_net(obs)
            pred_w = int(np.asarray(pred.cpu()).ravel().shape[0])
            assert pred_w == gsp_slot, (
                f"K={K}: head prediction width {pred_w} != actor slot {gsp_slot}"
            )
