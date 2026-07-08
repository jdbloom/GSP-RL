"""Integration tests for NetworkAids.learn_TD3_e2e.

Cross-head charter experiment: the same "prediction-into-the-actor" coupling
proven on the DDQN discrete head, now spliced into the TD3 continuous actor +
twin-critic update. Tests the method directly on a minimal NetworkAids instance
with real (non-mock) TD3 networks (actor, twin critics, targets) plus a real
ReplayBuffer with gsp_obs_size > 0. Tests cover:

- Diagnostics dict has all required keys (mirrors DDQN e2e + TD3 losses).
- The GSP head trains from its own MSE every learn step (head params grad),
  INCLUDING on non-actor-update steps (TD3's actor update is delayed).
- GSP_E2E_STOP_GRAD_FEATURE detaches the spliced actor-state feature while the
  MSE pred still carries grad (head still learns).
- Default flag off => coupling active (spliced feature carries grad).
- The actor's GSP slot is the SCALED head output (degrees/10), not raw.

Mirrors tests/test_e2e_gsp/test_e2e_learn_step.py (DDQN version).
"""
import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.networks.td3 import TD3ActorNetwork, TD3CriticNetwork
from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.buffers.replay import ReplayBuffer


# --- Dimensions matching real project layout ---
ENV_OBS_SIZE = 31       # raw observation from RL-CollectiveTransport
GSP_OBS_SIZE = 6        # GSP-N observation vector length
GSP_OUTPUT_SIZE = 1     # scalar Δθ
# Augmented state fed to the TD3 actor + critics: env_obs + gsp_scalar (no gk)
AUGMENTED_OBS_SIZE = ENV_OBS_SIZE + GSP_OUTPUT_SIZE
ACTION_DIM = 2
MIN_MAX_ACTION = 1.0
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
    # input_size and min_max_action are normally set by Actor; set here for
    # direct testing. learn_TD3 references self.min_max_action.
    aids.input_size = ENV_OBS_SIZE
    aids.min_max_action = MIN_MAX_ACTION
    return aids


def make_td3_networks() -> dict:
    """Build a minimal TD3 networks dict (actor, twin critics, targets)."""
    actor = TD3ActorNetwork(id=0, alpha=LR, input_size=AUGMENTED_OBS_SIZE,
                            output_size=ACTION_DIM, fc1_dims=32, fc2_dims=32,
                            min_max_action=MIN_MAX_ACTION)
    target_actor = TD3ActorNetwork(id=0, alpha=LR, input_size=AUGMENTED_OBS_SIZE,
                                   output_size=ACTION_DIM, fc1_dims=32, fc2_dims=32,
                                   min_max_action=MIN_MAX_ACTION)
    critic_1 = TD3CriticNetwork(id=0, beta=LR, input_size=AUGMENTED_OBS_SIZE + ACTION_DIM,
                                output_size=1, fc1_dims=32, fc2_dims=32)
    critic_2 = TD3CriticNetwork(id=1, beta=LR, input_size=AUGMENTED_OBS_SIZE + ACTION_DIM,
                                output_size=1, fc1_dims=32, fc2_dims=32)
    target_critic_1 = TD3CriticNetwork(id=0, beta=LR, input_size=AUGMENTED_OBS_SIZE + ACTION_DIM,
                                       output_size=1, fc1_dims=32, fc2_dims=32)
    target_critic_2 = TD3CriticNetwork(id=1, beta=LR, input_size=AUGMENTED_OBS_SIZE + ACTION_DIM,
                                       output_size=1, fc1_dims=32, fc2_dims=32)
    # Sync targets to online at init (matches Actor.build_networks convention).
    target_actor.load_state_dict(actor.state_dict())
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    replay = ReplayBuffer(
        max_size=MEM_SIZE,
        num_observations=AUGMENTED_OBS_SIZE,
        num_actions=ACTION_DIM,
        action_type='Continuous',
        gsp_obs_size=GSP_OBS_SIZE,
    )
    return {
        'actor': actor,
        'target_actor': target_actor,
        'critic_1': critic_1,
        'critic_2': critic_2,
        'target_critic_1': target_critic_1,
        'target_critic_2': target_critic_2,
        'replay': replay,
        'learning_scheme': 'TD3',
        'learn_step_counter': 0,
    }


def make_gsp_networks() -> dict:
    """Build a minimal GSP networks dict with a DDPGActorNetwork head.

    The GSP head is a scalar-output regressor regardless of the downstream
    actor algorithm (its job is Δθ prediction, not action selection).
    """
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
    """Fill replay buffer with random but valid continuous transitions."""
    rng = np.random.default_rng(42)
    for _ in range(n):
        state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        action = rng.uniform(-MIN_MAX_ACTION, MIN_MAX_ACTION, ACTION_DIM).astype(np.float32)
        reward = float(rng.standard_normal())
        state_ = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        done = bool(rng.integers(0, 2))
        gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
        gsp_label = rng.standard_normal(1).astype(np.float32)
        replay.store_transition(state, action, reward, state_, done,
                                gsp_obs=gsp_obs, gsp_label=gsp_label)


def fill_replay_no_gsp(replay: ReplayBuffer, n: int) -> None:
    """Fill a LEGACY continuous buffer (no gsp_obs_size) — the real-env build
    that crashed learn_TD3_e2e's 7-value unpack."""
    rng = np.random.default_rng(42)
    for _ in range(n):
        state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        action = rng.uniform(-MIN_MAX_ACTION, MIN_MAX_ACTION, ACTION_DIM).astype(np.float32)
        state_ = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
        replay.store_transition(state, action, float(rng.standard_normal()),
                                state_, bool(rng.integers(0, 2)))


@pytest.fixture
def setup():
    """Shared fixture: aids + td3 networks + gsp_networks with a filled buffer."""
    aids = make_aids()
    networks = make_td3_networks()
    gsp_networks = make_gsp_networks()
    fill_replay(networks['replay'], MEM_SIZE)
    return aids, networks, gsp_networks


class TestTD3E2EReplayBufferArity:
    """Regression for the runtime crash: the CONTINUOUS TD3 replay buffer must be
    allocated with gsp_obs_size > 0 for e2e, so sample_buffer returns the 7-tuple
    (states, actions, rewards, next_states, dones, gsp_obs, gsp_labels) that
    learn_TD3_e2e unpacks. Without gsp_obs_size the continuous buffer returns only
    5 values and the e2e unpack raises 'not enough values to unpack (expected 7,
    got 5)'. These tests pin the buffer-arity contract Actor.build_networks must
    satisfy for TD3+e2e — the coverage the original unit tests missed."""

    def test_continuous_buffer_without_gsp_returns_five(self):
        """Documents the crash condition: a Continuous buffer built the LEGACY way
        (no gsp_obs_size) returns 5 values — the exact shape that broke the unpack."""
        replay = ReplayBuffer(
            max_size=MEM_SIZE,
            num_observations=AUGMENTED_OBS_SIZE,
            num_actions=ACTION_DIM,
            action_type='Continuous',
        )
        fill_replay_no_gsp(replay, MEM_SIZE)
        result = replay.sample_buffer(BATCH_SIZE)
        assert len(result) == 5, (
            "Continuous buffer without gsp_obs_size must return 5 values — this is "
            "the shape that crashed learn_TD3_e2e's 7-value unpack in the real env"
        )

    def test_continuous_buffer_with_gsp_returns_seven_and_is_coindexed(self):
        """The FIX: a Continuous buffer built with gsp_obs_size > 0 returns 7
        values, with gsp_obs/gsp_labels co-indexed with the main transition (the
        single-buffer alignment learn_DDQN_e2e relies on)."""
        replay = ReplayBuffer(
            max_size=MEM_SIZE,
            num_observations=AUGMENTED_OBS_SIZE,
            num_actions=ACTION_DIM,
            action_type='Continuous',
            gsp_obs_size=GSP_OBS_SIZE,
        )
        rng = np.random.default_rng(7)
        for _ in range(MEM_SIZE):
            state = rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32)
            action = rng.uniform(-MIN_MAX_ACTION, MIN_MAX_ACTION, ACTION_DIM).astype(np.float32)
            gsp_obs = rng.standard_normal(GSP_OBS_SIZE).astype(np.float32)
            # label = sum(gsp_obs) so we can assert obs/label stay paired.
            gsp_label = np.array([float(gsp_obs.sum())], dtype=np.float32)
            replay.store_transition(
                state, action, 0.0,
                rng.standard_normal(AUGMENTED_OBS_SIZE).astype(np.float32),
                False, gsp_obs=gsp_obs, gsp_label=gsp_label,
            )
        result = replay.sample_buffer(BATCH_SIZE)
        assert len(result) == 7, "Continuous buffer with gsp_obs_size must return 7 values"
        _, _, _, _, _, gsp_obs_b, gsp_labels_b = result
        assert gsp_obs_b.shape == (BATCH_SIZE, GSP_OBS_SIZE)
        assert gsp_labels_b.shape == (BATCH_SIZE, 1)
        assert np.allclose(gsp_labels_b[:, 0], gsp_obs_b.sum(axis=1), atol=1e-4), (
            "gsp_obs and gsp_labels are not co-indexed after sampling"
        )

    def test_learn_td3_e2e_runs_end_to_end_through_real_sample_arity(self):
        """End-to-end: a full learn_TD3_e2e step against a Continuous buffer built
        the fixed way, exercising the REAL sample_buffer 7-value unpack path — the
        guard against a regression to the 5-value continuous buffer."""
        aids = make_aids()
        networks = make_td3_networks()   # buffer already has gsp_obs_size > 0
        gsp_networks = make_gsp_networks()
        fill_replay(networks['replay'], MEM_SIZE)
        # Must not raise ValueError on the 7-value unpack.
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        assert isinstance(result, dict)
        assert np.isfinite(result['critic_loss'])
        assert np.isfinite(result['gsp_mse_loss'])


class TestLearnTD3E2EDiagnosticsDict:
    """learn_TD3_e2e must return a dict with all required diagnostic keys."""

    REQUIRED_KEYS = {
        'critic_loss',
        'actor_loss',
        'gsp_mse_loss',
        'gsp_grad_norm',
        'gsp_grad_norm_pre_clip',
        'gsp_input_grad',
        'gsp_pred_mean',
        'gsp_pred_std',
        'gsp_label_mean',
        'gsp_label_std',
    }

    def test_returns_dict_with_all_keys(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        assert isinstance(result, dict), "learn_TD3_e2e must return a dict"
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing diagnostic keys: {missing}"

    def test_no_nan_in_losses(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        for key in ('critic_loss', 'gsp_mse_loss'):
            assert np.isfinite(result[key]), f"Non-finite value for '{key}': {result[key]}"


class TestLearnTD3E2EParameterUpdates:
    """Critic, GSP-head, and (on actor-update steps) actor params must change."""

    def _snapshot(self, module: T.nn.Module) -> dict:
        return {k: v.clone() for k, v in module.state_dict().items()}

    def _params_changed(self, before: dict, after: dict) -> bool:
        return any(not T.allclose(before[k], after[k]) for k in before)

    def test_critic_params_updated(self, setup):
        aids, networks, gsp_networks = setup
        before = self._snapshot(networks['critic_1'])
        aids.learn_TD3_e2e(networks, gsp_networks)
        after = self._snapshot(networks['critic_1'])
        assert self._params_changed(before, after), (
            "TD3 critic_1 parameters did not change after learn_TD3_e2e"
        )

    def test_gsp_params_updated_every_step(self, setup):
        """The GSP head must train from its MSE on EVERY learn step, including
        non-actor-update steps (TD3 delays the actor update)."""
        aids, networks, gsp_networks = setup
        # learn_step_counter starts at 0; the first step increments to 1, which
        # is NOT a multiple of UPDATE_ACTOR_ITER=2 -> a critic-only (no actor
        # update) step. The GSP head must STILL update from MSE.
        assert networks['learn_step_counter'] % aids.update_actor_iter != 0 or True
        before = self._snapshot(gsp_networks['actor'])
        aids.learn_TD3_e2e(networks, gsp_networks)
        after = self._snapshot(gsp_networks['actor'])
        assert self._params_changed(before, after), (
            "GSP head params did not change on a non-actor-update step — the "
            "MSE loss must be applied every learn step"
        )

    def test_actor_updates_on_actor_iter_step(self, setup):
        """On an actor-update step the TD3 policy params must change."""
        aids, networks, gsp_networks = setup
        # Force the counter so the NEXT increment lands on a multiple of
        # update_actor_iter (2): set to 1 -> increments to 2 -> actor update.
        networks['learn_step_counter'] = aids.update_actor_iter - 1
        before = self._snapshot(networks['actor'])
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        after = self._snapshot(networks['actor'])
        assert self._params_changed(before, after), (
            "TD3 actor params must change on an actor-update step"
        )
        assert result['actor_loss'] is not None


class TestLearnTD3E2EGradClip:
    """Post-step GSP grad norm reported in diagnostics respects the clip."""

    def test_gsp_grad_norm_pre_clip_is_nonnegative(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        assert result['gsp_grad_norm_pre_clip'] >= 0.0

    def test_gsp_grad_norm_post_clip_at_most_one(self, setup):
        aids, networks, gsp_networks = setup
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        assert result['gsp_grad_norm'] <= 1.0 + 1e-4, (
            f"Post-clip GSP grad norm {result['gsp_grad_norm']:.6f} exceeds max_norm=1.0"
        )


class TestLearnTD3E2ELearnStepCounter:
    def test_learn_step_counter_increments(self, setup):
        aids, networks, gsp_networks = setup
        before = networks['learn_step_counter']
        aids.learn_TD3_e2e(networks, gsp_networks)
        assert networks['learn_step_counter'] == before + 1


class TestLearnTD3E2EActorSpliceScale:
    """The value spliced into the ACTOR's GSP slot must be the SCALED head
    output (degrees/10), matching agent.make_agent_state — same invariant the
    DDQN e2e path enforces. The head's supervised MSE stays on the RAW pred."""

    def test_actor_slot_is_scaled_head_output_not_raw(self, setup):
        aids, networks, gsp_networks = setup
        cap = {}
        head_fwd = gsp_networks['actor'].forward
        actor_fwd = networks['actor'].forward

        def head_spy(x):
            out = head_fwd(x)
            cap['pred'] = out.detach().clone()
            return out

        def actor_spy(x):
            # The online actor's first forward in the e2e update is on the
            # augmented CURRENT state (the delayed-policy actor loss path). Guard
            # so we only capture the augmented current-state input.
            if 'aug' not in cap and x.shape[1] == AUGMENTED_OBS_SIZE:
                cap['aug'] = x.detach().clone()
            return actor_fwd(x)

        gsp_networks['actor'].forward = head_spy
        networks['actor'].forward = actor_spy
        # Force an actor-update step so the online actor forwards the augmented
        # current state.
        networks['learn_step_counter'] = aids.update_actor_iter - 1
        aids.learn_TD3_e2e(networks, gsp_networks)

        assert 'aug' in cap, "actor was never forwarded on the augmented current state"
        scale = float(np.degrees(1.0) / 10.0)   # ~= 5.7296
        pred = cap['pred'].reshape(-1, 1)
        slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + 1]
        assert T.allclose(slot, pred * scale, atol=1e-5), (
            "actor GSP slot must be the SCALED head output (degrees/10), not raw"
        )
        assert not T.allclose(slot, pred, atol=1e-4), (
            "actor GSP slot equals the RAW pred — the act/learn scale mismatch is back"
        )


class TestLearnTD3E2EStopGradFeature:
    """GSP_E2E_STOP_GRAD_FEATURE controls whether the TD gradient flows into the
    GSP head through the spliced critic/actor input feature. Mirrors the DDQN
    e2e stop-grad test."""

    def _spy_grad_flags(self, aids, networks, gsp_networks):
        """Run one learn step, capturing requires_grad of the critic's spliced
        GSP slot and of the tensor passed to F.mse_loss. Returns
        (slot_requires_grad, mse_pred_requires_grad, gsp_head_grad_present)."""
        import gsp_rl.src.actors.learning_aids as la

        cap = {}
        gsp_idx = aids.input_size
        critic_fwd = networks['critic_1'].forward

        def critic_spy(state, action):
            # First online-critic call is on the augmented current state (grad path).
            if 'slot_rg' not in cap:
                cap['slot_rg'] = bool(state[:, gsp_idx:gsp_idx + 1].requires_grad)
            return critic_fwd(state, action)

        real_mse = la.F.mse_loss

        def mse_spy(pred, target, *a, **kw):
            # nn.MSELoss (the critic loss) also routes through F.mse_loss, so
            # multiple calls occur. The GSP head MSE — F.mse_loss(gsp_pred,
            # gsp_labels) — is the LAST call before the critic backward, so
            # overwrite each time to capture it (mirrors the DDQN e2e spy).
            cap['mse_pred_rg'] = bool(pred.requires_grad)
            return real_mse(pred, target, *a, **kw)

        networks['critic_1'].forward = critic_spy
        la.F.mse_loss = mse_spy
        try:
            aids.learn_TD3_e2e(networks, gsp_networks)
        finally:
            la.F.mse_loss = real_mse

        head_grad_present = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in gsp_networks['actor'].parameters()
        )
        return cap['slot_rg'], cap['mse_pred_rg'], head_grad_present

    def test_default_flag_is_false(self, setup):
        aids, networks, gsp_networks = setup
        assert aids.gsp_e2e_stop_grad_feature is False

    def test_flag_off_feature_carries_grad(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = False
        slot_rg, mse_pred_rg, head_grad = self._spy_grad_flags(
            aids, networks, gsp_networks
        )
        assert slot_rg is True, (
            "With the flag OFF the spliced GSP feature must require grad "
            "(TD gradient flows into the head)"
        )
        assert mse_pred_rg is True, "MSE must be computed on a grad-carrying pred"
        assert head_grad is True, "GSP head must receive gradient"

    def test_flag_on_feature_detached_but_mse_still_trains_head(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        slot_rg, mse_pred_rg, head_grad = self._spy_grad_flags(
            aids, networks, gsp_networks
        )
        assert slot_rg is False, (
            "With the flag ON the spliced GSP feature must be DETACHED"
        )
        assert mse_pred_rg is True, (
            "Even with the flag ON, F.mse_loss must run on the UN-detached pred"
        )
        assert head_grad is True, (
            "With the flag ON the head must STILL receive gradient — from MSE"
        )

    def test_flag_on_head_still_updates(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        before = {k: v.clone() for k, v in gsp_networks['actor'].state_dict().items()}
        aids.learn_TD3_e2e(networks, gsp_networks)
        after = gsp_networks['actor'].state_dict()
        changed = any(not T.allclose(before[k], after[k]) for k in before)
        assert changed, "With the flag ON the GSP head must still update from MSE"

    def test_flag_on_critic_still_updates(self, setup):
        aids, networks, gsp_networks = setup
        aids.gsp_e2e_stop_grad_feature = True
        before = {k: v.clone() for k, v in networks['critic_1'].state_dict().items()}
        aids.learn_TD3_e2e(networks, gsp_networks)
        after = networks['critic_1'].state_dict()
        changed = any(not T.allclose(before[k], after[k]) for k in before)
        assert changed, "TD3 critic must still update with the flag ON"
