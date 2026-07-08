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


class TestLearnDDQNE2EActorSpliceScale:
    """Regression: the value spliced into the ACTOR's GSP slot must be scaled to
    match agent.make_agent_state (RL-CT), which writes degrees(pred/10) =
    pred * (180/pi/10) ~= pred * 5.7296. The learn path previously spliced the RAW
    head output, so the Q-net was trained on a ~5.73x-smaller value than it saw at
    act-time and than the stored next-state slot — an internally inconsistent
    Bellman update that stopped the E2E actor learning the task. The head's
    supervised MSE, by contrast, must stay on the RAW prediction."""

    def test_actor_slot_is_scaled_head_output_not_raw(self, setup):
        aids, networks, gsp_networks = setup
        cap = {}
        head_fwd = gsp_networks['actor'].forward
        q_fwd = networks['q_eval'].forward

        def head_spy(x):
            out = head_fwd(x)
            cap['pred'] = out.detach().clone()
            return out

        def q_spy(x):
            if 'aug' not in cap:            # first q_eval call is on the augmented current state
                cap['aug'] = x.detach().clone()
            return q_fwd(x)

        gsp_networks['actor'].forward = head_spy
        networks['q_eval'].forward = q_spy
        aids.learn_DDQN_e2e(networks, gsp_networks)

        scale = float(np.degrees(1.0) / 10.0)   # == degrees(x/10)/x ~= 5.7296
        pred = cap['pred'].reshape(-1, 1)        # (batch, 1) raw head output
        slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + 1]  # the actor's GSP feature
        assert T.allclose(slot, pred * scale, atol=1e-5), (
            "actor GSP slot must be the SCALED head output (degrees/10), not raw"
        )
        # And it must NOT be the raw prediction (guard against a silent revert).
        assert not T.allclose(slot, pred, atol=1e-4), (
            "actor GSP slot equals the RAW pred — the act/learn scale mismatch is back"
        )


class TestLearnDDQNE2EStopGradFeature:
    """GSP_E2E_STOP_GRAD_FEATURE (default False) controls whether the actor's TD
    gradient is allowed to flow back into the GSP head through the spliced Q-input
    feature.

    Motivation: the head is trained through BOTH its supervised MSE loss AND the
    actor's TD gradient flowing back through the spliced feature. That coupling
    makes the feature non-stationary and prevents the critic from converging. With
    the flag ON we DETACH the prediction where it enters the actor's Q-input, so
    the actor's TD gradient no longer perturbs the head — the head then trains ONLY
    from its own MSE loss (which must stay attached to the un-detached prediction).

    The cleanest observable check: spy on the tensor spliced into the actor's
    Q-input (the augmented state) and on the tensor fed to F.mse_loss:
      * flag ON  -> spliced feature is detached (requires_grad == False)
      * flag OFF -> spliced feature carries grad (requires_grad == True)
      * MSE input requires grad in BOTH cases (head still learns from MSE)
    And the flag defaults to False, preserving byte-identical prior behavior.
    """

    def _spy_grad_flags(self, aids, networks, gsp_networks):
        """Run one learn step, capturing requires_grad of the actor's GSP slot and
        the tensor passed to F.mse_loss. Returns (slot_requires_grad,
        mse_pred_requires_grad, gsp_head_grad_present)."""
        import gsp_rl.src.actors.learning_aids as la

        cap = {}
        gsp_idx = aids.input_size
        q_fwd = networks['q_eval'].forward

        def q_spy(x):
            # First q_eval call is on the augmented current state (grad path).
            if 'slot_rg' not in cap:
                # The spliced GSP feature column of the augmented state.
                cap['slot_rg'] = bool(x[:, gsp_idx:gsp_idx + 1].requires_grad)
            return q_fwd(x)

        real_mse = la.F.mse_loss

        def mse_spy(pred, target, *a, **kw):
            cap['mse_pred_rg'] = bool(pred.requires_grad)
            return real_mse(pred, target, *a, **kw)

        networks['q_eval'].forward = q_spy
        la.F.mse_loss = mse_spy
        try:
            aids.learn_DDQN_e2e(networks, gsp_networks)
        finally:
            la.F.mse_loss = real_mse

        # After the step, the GSP head must have received gradient (from the MSE
        # loss even when the actor-side feature is detached). We can only observe
        # this indirectly: params changed. Snapshot-based check is done separately;
        # here we confirm at least one head param carries a grad after backward.
        head_grad_present = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in gsp_networks['actor'].parameters()
        )
        return cap['slot_rg'], cap['mse_pred_rg'], head_grad_present

    def test_default_flag_is_false(self, setup):
        aids, networks, gsp_networks = setup
        assert aids.gsp_e2e_stop_grad_feature is False, (
            "GSP_E2E_STOP_GRAD_FEATURE must default to False"
        )

    def test_flag_off_actor_feature_carries_grad(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = False
        slot_rg, mse_pred_rg, head_grad = self._spy_grad_flags(
            aids, networks, gsp_networks
        )
        assert slot_rg is True, (
            "With the flag OFF the actor's spliced GSP feature must require grad "
            "(TD gradient flows into the head) — preserves prior behavior"
        )
        assert mse_pred_rg is True, "MSE must be computed on a grad-carrying pred"
        assert head_grad is True, "GSP head must receive gradient"

    def test_flag_on_actor_feature_is_detached_but_mse_still_trains_head(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        slot_rg, mse_pred_rg, head_grad = self._spy_grad_flags(
            aids, networks, gsp_networks
        )
        assert slot_rg is False, (
            "With the flag ON the actor's spliced GSP feature must be DETACHED "
            "(requires_grad == False) so the TD gradient cannot perturb the head"
        )
        assert mse_pred_rg is True, (
            "Even with the flag ON, F.mse_loss must run on the UN-detached pred so "
            "the head still learns to predict from its own supervised loss"
        )
        assert head_grad is True, (
            "With the flag ON the head must STILL receive gradient — from the MSE "
            "loss — even though the actor TD gradient is severed"
        )

    def test_flag_on_head_still_updates(self, setup):
        """End-to-end: with the flag ON the GSP head params must still change,
        driven purely by the MSE loss."""
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        before = {k: v.clone() for k, v in gsp_networks['actor'].state_dict().items()}
        aids.learn_DDQN_e2e(networks, gsp_networks)
        after = gsp_networks['actor'].state_dict()
        changed = any(not T.allclose(before[k], after[k]) for k in before)
        assert changed, (
            "With the flag ON the GSP head must still update from its MSE loss"
        )

    def test_flag_on_ddqn_still_updates(self, setup):
        """The actor (DDQN) must still learn with the flag ON — only the gradient
        path INTO the head is cut, not the actor's own updates."""
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        before = {k: v.clone() for k, v in networks['q_eval'].state_dict().items()}
        aids.learn_DDQN_e2e(networks, gsp_networks)
        after = networks['q_eval'].state_dict()
        changed = any(not T.allclose(before[k], after[k]) for k in before)
        assert changed, "DDQN q_eval must still update with the flag ON"

    def test_config_key_read_from_constructor(self):
        """The flag is read from the SAME config source as GSP_E2E_LAMBDA, under
        the key GSP_E2E_STOP_GRAD_FEATURE."""
        cfg = dict(MINIMAL_CONFIG)
        cfg['GSP_E2E_STOP_GRAD_FEATURE'] = True
        aids = NetworkAids(cfg)
        assert aids.gsp_e2e_stop_grad_feature is True, (
            "NetworkAids must read GSP_E2E_STOP_GRAD_FEATURE from config"
        )
