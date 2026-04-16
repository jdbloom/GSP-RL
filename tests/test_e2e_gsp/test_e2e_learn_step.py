"""Integration tests for NetworkAids.learn_DDQN_e2e.

Tests the learn_DDQN_e2e method directly on a minimal NetworkAids instance
with real (non-mock) DDQN and GSP networks plus a real ReplayBuffer with
gsp_obs_size > 0. Tests cover:
- Diagnostics dict has all required keys
- Both optimizer parameter updates actually occur (param tensors change)
- GSP gradients are clipped (post-clip norm <= 1.0 + tolerance)
- No NaN in loss values
- Combined loss equals ddqn_loss + lambda * gsp_mse_loss

Note: Task 3 (wiring learn_DDQN_e2e into Actor.learn()) is not yet done.
We test the method directly via NetworkAids instantiation.
"""
import numpy as np
import pytest
import copy
import torch as T

from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.buffers.replay import ReplayBuffer


# --- Dimensions matching real project layout ---
ENV_OBS_SIZE = 31       # raw observation from RL-CollectiveTransport
GSP_OBS_SIZE = 6        # GSP-N observation vector length
GSP_OUTPUT_SIZE = 1     # scalar Δθ
# Augmented state fed to DDQN: env_obs + gsp_scalar + (no gk for these tests)
AUGMENTED_OBS_SIZE = ENV_OBS_SIZE + GSP_OUTPUT_SIZE
NUM_ACTIONS = 5
BATCH_SIZE = 16
MEM_SIZE = 200
LR = 1e-3


MINIMAL_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": LR,
    "BETA": LR,
    "LR": LR,
    "EPSILON": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DEC": 0.001,
    "BATCH_SIZE": BATCH_SIZE,
    "MEM_SIZE": MEM_SIZE,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.1,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": BATCH_SIZE,
}


def make_aids() -> NetworkAids:
    """Create a bare NetworkAids instance (no Actor overhead)."""
    aids = NetworkAids(MINIMAL_CONFIG)
    # input_size is normally set by Actor; set it here for direct testing
    aids.input_size = ENV_OBS_SIZE
    return aids


def make_ddqn_networks() -> dict:
    """Build a minimal DDQN networks dict."""
    q_eval = DDQN(id=0, lr=LR, input_size=AUGMENTED_OBS_SIZE, output_size=NUM_ACTIONS,
                  fc1_dims=32, fc2_dims=32)
    q_next = DDQN(id=0, lr=LR, input_size=AUGMENTED_OBS_SIZE, output_size=NUM_ACTIONS,
                  fc1_dims=32, fc2_dims=32)
    replay = ReplayBuffer(
        max_size=MEM_SIZE,
        num_observations=AUGMENTED_OBS_SIZE,
        num_actions=1,
        action_type='Discrete',
        gsp_obs_size=GSP_OBS_SIZE,
    )
    return {
        'q_eval': q_eval,
        'q_next': q_next,
        'replay': replay,
        'learning_scheme': 'DDQN',
        'learn_step_counter': 0,
    }


def make_gsp_networks() -> dict:
    """Build a minimal GSP networks dict with a DDPGActorNetwork head."""
    actor = DDPGActorNetwork(
        id=0,
        lr=LR,
        input_size=GSP_OBS_SIZE,
        output_size=GSP_OUTPUT_SIZE,
        fc1_dims=32,
        fc2_dims=16,
        min_max_action=1.0,
        use_linear_output=True,
    )
    return {
        'actor': actor,
        'learning_scheme': 'DDPG',
        'learn_step_counter': 0,
    }


def fill_replay(replay: ReplayBuffer, n: int) -> None:
    """Fill replay buffer with random but valid transitions."""
    rng = np.random.default_rng(42)
    for _ in range(n):
        state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        action = int(rng.integers(0, NUM_ACTIONS))
        reward = float(rng.standard_normal())
        state_ = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        done = bool(rng.integers(0, 2))
        gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
        gsp_label = rng.standard_normal(1).astype(np.float32)
        replay.store_transition(state, action, reward, state_, done,
                                gsp_obs=gsp_obs, gsp_label=gsp_label)


@pytest.fixture
def setup():
    """Shared fixture: aids + networks + gsp_networks with a filled replay buffer."""
    aids = make_aids()
    networks = make_ddqn_networks()
    gsp_networks = make_gsp_networks()
    fill_replay(networks['replay'], MEM_SIZE)
    return aids, networks, gsp_networks


class TestLearnDDQNE2EDiagnosticsDict:
    """learn_DDQN_e2e must return a dict with all required diagnostic keys."""

    REQUIRED_KEYS = {
        'ddqn_loss',
        'gsp_mse_loss',
        'total_loss',
        'gsp_grad_norm',
        'gsp_grad_norm_pre_clip',
        'ddqn_grad_norm',
        'gsp_input_grad',
        'gsp_pred_mean',
        'gsp_pred_std',
        'gsp_label_mean',
        'gsp_label_std',
    }

    def test_returns_dict_with_all_keys(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        assert isinstance(result, dict), "learn_DDQN_e2e must return a dict"
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing diagnostic keys: {missing}"

    def test_no_nan_in_losses(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        for key in ('ddqn_loss', 'gsp_mse_loss', 'total_loss'):
            assert np.isfinite(result[key]), f"Non-finite value for '{key}': {result[key]}"

    def test_total_loss_equals_ddqn_plus_lambda_times_gsp(self, setup):
        aids, networks, gsp_networks = setup
        # Default lambda is 1.0
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        expected = result['ddqn_loss'] + 1.0 * result['gsp_mse_loss']
        assert abs(result['total_loss'] - expected) < 1e-5, (
            f"total_loss {result['total_loss']:.6f} != "
            f"ddqn_loss + gsp_mse_loss = {expected:.6f}"
        )


class TestLearnDDQNE2EParameterUpdates:
    """Both the DDQN and GSP network parameters must change after a learn step."""

    def _snapshot(self, module: T.nn.Module) -> dict:
        return {k: v.clone() for k, v in module.state_dict().items()}

    def _params_changed(self, before: dict, after: dict) -> bool:
        return any(
            not T.allclose(before[k], after[k])
            for k in before
        )

    def test_ddqn_params_updated(self, setup):
        aids, networks, gsp_networks = setup
        before = self._snapshot(networks['q_eval'])
        aids.learn_DDQN_e2e(networks, gsp_networks)
        after = self._snapshot(networks['q_eval'])
        assert self._params_changed(before, after), (
            "DDQN q_eval parameters did not change after learn_DDQN_e2e"
        )

    def test_gsp_params_updated(self, setup):
        aids, networks, gsp_networks = setup
        before = self._snapshot(gsp_networks['actor'])
        aids.learn_DDQN_e2e(networks, gsp_networks)
        after = self._snapshot(gsp_networks['actor'])
        assert self._params_changed(before, after), (
            "GSP actor parameters did not change after learn_DDQN_e2e"
        )


class TestLearnDDQNE2EGradClip:
    """Post-step GSP grad norm reported in diagnostics must respect the clip threshold."""

    def test_gsp_grad_norm_pre_clip_is_nonnegative(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        assert result['gsp_grad_norm_pre_clip'] >= 0.0, (
            "Pre-clip GSP grad norm must be non-negative"
        )

    def test_gsp_grad_norm_post_clip_at_most_one(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        # Post-clip norm is measured after clip_grad_norm_ but before optimizer.step()
        # so it should be <= max_norm (1.0) + small floating-point tolerance
        assert result['gsp_grad_norm'] <= 1.0 + 1e-4, (
            f"Post-clip GSP grad norm {result['gsp_grad_norm']:.6f} exceeds max_norm=1.0"
        )


class TestLearnDDQNE2ELambdaScaling:
    """gsp_e2e_lambda attribute on aids scales the GSP auxiliary loss."""

    def test_lambda_zero_makes_gsp_loss_not_affect_total(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_lambda = 0.0
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        # With lambda=0, total_loss == ddqn_loss (ignoring the gsp_mse term)
        assert abs(result['total_loss'] - result['ddqn_loss']) < 1e-5, (
            f"With lambda=0, total_loss should equal ddqn_loss. "
            f"Got total={result['total_loss']:.6f}, ddqn={result['ddqn_loss']:.6f}"
        )

    def test_lambda_two_doubles_gsp_contribution(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_lambda = 2.0
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        expected = result['ddqn_loss'] + 2.0 * result['gsp_mse_loss']
        assert abs(result['total_loss'] - expected) < 1e-5, (
            f"With lambda=2.0, total_loss should be ddqn + 2*gsp_mse. "
            f"Got {result['total_loss']:.6f}, expected {expected:.6f}"
        )


class TestLearnDDQNE2ELearnStepCounter:
    """learn_step_counter in networks dict must increment after each learn step."""

    def test_learn_step_counter_increments(self, setup):
        aids, networks, gsp_networks = setup
        before = networks['learn_step_counter']
        aids.learn_DDQN_e2e(networks, gsp_networks)
        assert networks['learn_step_counter'] == before + 1
