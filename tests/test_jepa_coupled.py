"""Tests for the coupled-JEPA fix (2026-07-05 literature-convergent).

Covers the three flags and the diagnostics/ablation hook:

  1. GSP_JEPA_COUPLE_VALUE          — DDQN value-loss gradient reaches the
     online encoder (learn_DDQN_jepa_coupled), mirroring learn_DDQN_e2e.
  1b. GSP_JEPA_VALUE_STOPGRAD_ACTOR — the explicit Ni-couple vs Dreamer-freeze
     resolution: when True the spliced latent is detached before the Q-net, so
     the value loss does NOT reach the encoder (encoder shaped only by
     self-prediction); when False the value gradient flows fully into it.
  2. GSP_JEPA_ACTION_COND           — the coupled predictor consumes the real
     actor action a_t.
  3. GSP_JEPA_COSINE_LOSS           — the latent loss is cosine, not raw MSE.
  4. VICReg + EMA anti-collapse tunables fire.
  5. Diagnostics latent_rank/latent_var/jepa_pred_mse are returned.

All flags default OFF; an actor built with only GSP_JEPA_ENABLED trains via the
legacy uncoupled learn_gsp_jepa path unchanged.
"""
import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


ENV_OBS = 4
GSP_INPUT = 6
ENC_DIM = 8
NUM_ACTIONS = 3
BATCH = 16


def base_config(**overrides):
    cfg = {
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.002,
        "LR": 1e-3,
        "EPSILON": 0.0,
        "EPS_MIN": 0.0,
        "EPS_DEC": 0.0,
        "BATCH_SIZE": BATCH,
        "MEM_SIZE": 2000,
        "REPLACE_TARGET_COUNTER": 100,
        "NOISE": 0.0,
        "UPDATE_ACTOR_ITER": 2,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 1,
        "GSP_BATCH_SIZE": BATCH,
        "GSP_JEPA_ENABLED": True,
        "GSP_ENCODER_DIM": ENC_DIM,
    }
    cfg.update(overrides)
    return cfg


def make_actor(**cfg_overrides):
    cfg = base_config(**cfg_overrides)
    actor = Actor(
        id=0,
        config=cfg,
        network="DDQN",
        input_size=ENV_OBS,
        output_size=NUM_ACTIONS,
        min_max_action=1,
        meta_param_size=0,
        gsp=True,
        gsp_input_size=GSP_INPUT,
        gsp_output_size=1,
    )
    return actor


def fill_main_replay(actor, n=BATCH * 2):
    """Store n transitions on the main replay with gsp_obs (required for coupling)."""
    aug_size = actor.network_input_size  # ENV_OBS + ENC_DIM
    for _ in range(n):
        s = np.random.randn(aug_size).astype(np.float32)
        s_ = np.random.randn(aug_size).astype(np.float32)
        a = np.random.randint(0, NUM_ACTIONS)
        gsp_obs = np.random.randn(GSP_INPUT).astype(np.float32)
        actor.store_agent_transition(
            s, a, 0.5, s_, False, gsp_obs=gsp_obs, gsp_label=np.zeros(1, np.float32)
        )


class TestCoupleValueGradient:
    """GSP_JEPA_COUPLE_VALUE: value loss must reach the online encoder."""

    def test_couple_flag_builds_main_replay_with_gsp_obs(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=True)
        assert actor.networks["replay"].gsp_obs_size == GSP_INPUT, (
            "Coupled JEPA must allocate gsp_obs in the main replay"
        )

    def test_value_gradient_reaches_encoder_when_coupled(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=True)
        fill_main_replay(actor)
        # Zero encoder grads, run the coupled learn step, inspect grads.
        for p in actor.gsp_encoder_online.parameters():
            p.grad = None
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None
        grads = [
            p.grad for p in actor.gsp_encoder_online.parameters() if p.grad is not None
        ]
        assert len(grads) > 0, "No encoder params received a gradient"
        total = sum(g.abs().sum().item() for g in grads)
        assert total > 0, "Value loss did not flow into the online encoder"

    def test_stopgrad_actor_blocks_value_gradient_to_encoder(self):
        """With GSP_JEPA_VALUE_STOPGRAD_ACTOR=True the *value* loss must not
        reach the encoder — only self-prediction shapes it (Dreamer-freeze)."""
        actor = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_JEPA_VALUE_STOPGRAD_ACTOR=True,
            # Disable self-prediction contribution so any encoder grad would
            # have to come from the (blocked) value path.
            GSP_JEPA_SELFPRED_COEF=0.0,
        )
        fill_main_replay(actor)
        for p in actor.gsp_encoder_online.parameters():
            p.grad = None
        actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        grads = [
            p.grad for p in actor.gsp_encoder_online.parameters() if p.grad is not None
        ]
        total = sum(g.abs().sum().item() for g in grads) if grads else 0.0
        assert total == 0.0, (
            "Value gradient reached the encoder despite VALUE_STOPGRAD_ACTOR=True"
        )

    def test_coupled_returns_diagnostics(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=True)
        fill_main_replay(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        for key in ("latent_rank", "latent_var", "jepa_pred_mse"):
            assert key in result, f"coupled learn step missing diagnostic '{key}'"


class TestActionConditionedCoupling:
    """GSP_JEPA_ACTION_COND: the coupled predictor consumes the real action."""

    def test_predictor_is_action_conditioned(self):
        actor = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_JEPA_ACTION_COND=True,
            GSP_JEPA_ACTION_DIM=NUM_ACTIONS,
        )
        assert actor.gsp_predictor.action_dim == NUM_ACTIONS
        assert actor.gsp_predictor.fc1.weight.shape[1] == ENC_DIM + NUM_ACTIONS

    def test_coupled_action_conditioned_learn_step_runs(self):
        actor = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_JEPA_ACTION_COND=True,
            GSP_JEPA_ACTION_DIM=NUM_ACTIONS,
        )
        fill_main_replay(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None and np.isfinite(result["jepa_pred_mse"])


class TestCosineLossFlag:
    """GSP_JEPA_COSINE_LOSS: coupled step uses cosine latent loss."""

    def test_cosine_flag_learn_step_runs(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=True, GSP_JEPA_COSINE_LOSS=True)
        fill_main_replay(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None
        # cosine loss lives in [0, 2]
        assert 0.0 <= result["jepa_pred_mse"] <= 2.0 + 1e-4


class TestVicregAndEmaTunables:
    """VICReg var/cov and EMA target remain wired under coupling."""

    def test_vicreg_fires_in_coupled_step(self):
        actor = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_VICREG_ENABLED=True,
            GSP_VICREG_VAR_COEF=5.0,
            GSP_VICREG_COV_COEF=1.0,
        )
        assert actor.gsp_vicreg_enabled is True
        assert actor.gsp_vicreg_var_coef == 5.0
        fill_main_replay(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None and np.isfinite(result["total_loss"])

    def test_ema_tau_configurable(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=True, GSP_ENCODER_EMA_TAU=0.9)
        assert actor.gsp_encoder_ema_tau == 0.9
        fill_main_replay(actor)
        # Run one coupled step first so any device co-location happens, then
        # snapshot the target and run a second step to observe the EMA update
        # (avoids comparing pre/post-device-move tensors).
        actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        before = [p.detach().cpu().clone() for p in actor.gsp_encoder_target.parameters()]
        actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        after = [p.detach().cpu() for p in actor.gsp_encoder_target.parameters()]
        moved = any(not T.allclose(b, a) for b, a in zip(before, after))
        assert moved, "EMA target encoder did not update after coupled step"


class TestFlagsOffPreserveLegacy:
    """All new flags OFF → legacy uncoupled JEPA path is used unchanged."""

    def test_default_jepa_does_not_build_gsp_obs(self):
        actor = make_actor()  # only GSP_JEPA_ENABLED
        assert actor.networks["replay"].gsp_obs_size == 0, (
            "Uncoupled JEPA must not allocate gsp_obs (byte-identical to legacy)"
        )

    def test_default_predictor_not_action_conditioned(self):
        actor = make_actor()
        assert actor.gsp_predictor.action_dim == 0

    def test_learn_dispatches_to_legacy_when_not_coupled(self):
        actor = make_actor()
        assert actor.gsp_jepa_couple_value is False


class TestFlagsOffByteIdentical:
    """With all new flags OFF, encoder/predictor forward output is bit-identical
    to a plain GSP_JEPA_ENABLED build seeded the same way — the new flags cannot
    change legacy behavior."""

    def _build_seeded(self, seed, **overrides):
        T.manual_seed(seed)
        np.random.seed(seed)
        return make_actor(**overrides)

    def test_encoder_forward_identical_with_flags_off(self):
        a1 = self._build_seeded(1234)
        a2 = self._build_seeded(1234, GSP_JEPA_COUPLE_VALUE=False,
                                GSP_JEPA_ACTION_COND=False, GSP_JEPA_COSINE_LOSS=False)
        x = T.randn(BATCH, GSP_INPUT).to(a1.gsp_encoder_online.device)
        with T.no_grad():
            z1 = a1.gsp_encoder_online(x)
            z2 = a2.gsp_encoder_online(x.to(a2.gsp_encoder_online.device))
        assert T.allclose(z1.cpu(), z2.cpu(), atol=0.0), (
            "Flags-off encoder forward differs from legacy build"
        )

    def test_predictor_shape_identical_with_flags_off(self):
        a = make_actor()  # flags off
        # Legacy predictor: fc1 input == encoder_dim (no action concat).
        assert a.gsp_predictor.fc1.weight.shape[1] == ENC_DIM
