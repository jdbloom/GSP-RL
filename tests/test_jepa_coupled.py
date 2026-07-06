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


class TestSkipStandaloneSelfPredWhenCoupled:
    """Under value-coupling, learn_gsp() must NOT also run the standalone
    learn_gsp_jepa: the coupled step (learn_DDQN_jepa_coupled) already trains the
    encoder's self-prediction, and the standalone path crashes under action-
    conditioning (its buffer stores no action). Regression for the ep-0 crash
    'JEPAPredictor built with action_dim > 0 requires an action tensor'."""

    def test_learn_gsp_skips_standalone_when_coupled(self):
        actor = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_JEPA_ACTION_COND=True,
            GSP_JEPA_ACTION_DIM=NUM_ACTIONS,
        )
        actor.gsp_networks["replay"].mem_ctr = BATCH * 2  # pass the buffer guard
        called = {"n": 0}
        actor.learn_gsp_jepa = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        actor.learn_gsp()
        assert called["n"] == 0, "standalone learn_gsp_jepa must be skipped when coupled"

    def test_learn_gsp_runs_standalone_when_not_coupled(self):
        actor = make_actor(GSP_JEPA_COUPLE_VALUE=False)
        actor.gsp_networks["replay"].mem_ctr = BATCH * 2
        called = {"n": 0}

        def _rec(*a, **k):
            called["n"] += 1
            return None

        actor.learn_gsp_jepa = _rec
        actor.learn_gsp()
        assert called["n"] == 1, "standalone learn_gsp_jepa must run when not coupled"


# ---------------------------------------------------------------------------
# Latent-primary actor head (GSP_ACTOR_LATENT_PRIMARY) +
# SimNorm (GSP_JEPA_SIMNORM) — 2026-07-06 pre-registration.
# ---------------------------------------------------------------------------
from gsp_rl.src.networks.jepa import simnorm, JEPAEncoder  # noqa: E402


def fill_main_replay_primary(actor, n=BATCH * 2):
    """Fill the main replay with augmented states of the actor's OWN Q-net input
    width (network_input_size), whatever the flags set it to."""
    aug_size = actor.network_input_size
    for _ in range(n):
        s = np.random.randn(aug_size).astype(np.float32)
        s_ = np.random.randn(aug_size).astype(np.float32)
        a = np.random.randint(0, NUM_ACTIONS)
        gsp_obs = np.random.randn(GSP_INPUT).astype(np.float32)
        actor.store_agent_transition(
            s, a, 0.5, s_, False, gsp_obs=gsp_obs, gsp_label=np.zeros(1, np.float32)
        )


class TestLatentPrimaryInputDim:
    """GSP_ACTOR_LATENT_PRIMARY drops the raw env-obs block from the Q-net input."""

    def test_default_concat_input_dim(self):
        """Flag OFF: Q-net input == env_obs + encoder_dim (unchanged legacy)."""
        actor = make_actor()  # GSP_ACTOR_LATENT_PRIMARY defaults False
        assert actor.gsp_actor_latent_primary is False
        assert actor.network_input_size == ENV_OBS + ENC_DIM
        # The Q-net's first Linear layer must match.
        assert actor.networks["q_eval"].fc1.weight.shape[1] == ENV_OBS + ENC_DIM

    def test_latent_primary_drops_env_obs(self):
        """Flag ON: Q-net input == encoder_dim only (env_obs dropped)."""
        actor = make_actor(GSP_ACTOR_LATENT_PRIMARY=True)
        assert actor.gsp_actor_latent_primary is True
        assert actor.network_input_size == ENC_DIM
        assert actor.networks["q_eval"].fc1.weight.shape[1] == ENC_DIM

    def test_latent_primary_coupled_learn_step_runs(self):
        """Coupled learn step must run coherently under latent-primary: the
        spliced state is [latent | (global)] with the latent slot at index 0,
        so the augmented Q-forward matches network_input_size."""
        actor = make_actor(
            GSP_ACTOR_LATENT_PRIMARY=True, GSP_JEPA_COUPLE_VALUE=True
        )
        fill_main_replay_primary(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None and np.isfinite(result["ddqn_loss"])

    def test_latent_primary_value_gradient_reaches_encoder(self):
        """Latent-primary + coupling: the value loss MUST reach the encoder
        (the whole point — the actor can only get env info via the latent)."""
        actor = make_actor(
            GSP_ACTOR_LATENT_PRIMARY=True, GSP_JEPA_COUPLE_VALUE=True
        )
        fill_main_replay_primary(actor)
        for p in actor.gsp_encoder_online.parameters():
            p.grad = None
        actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        grads = [
            p.grad for p in actor.gsp_encoder_online.parameters() if p.grad is not None
        ]
        total = sum(g.abs().sum().item() for g in grads) if grads else 0.0
        assert total > 0, "value loss did not reach encoder under latent-primary"

    def test_latent_primary_diagnostics_do_not_crash(self):
        """compute_diagnostics must run under latent-primary: the actor-usage
        pred_slice base is 0 (env_obs dropped), so the slice stays in-bounds and
        the diagnostic path does not raise (the NaN sentinel is acceptable when
        the latent IS the whole input)."""
        actor = make_actor(
            GSP_ACTOR_LATENT_PRIMARY=True,
            GSP_JEPA_COUPLE_VALUE=True,
            DIAGNOSTICS_ENABLED=True,
            DIAGNOSTICS_FREEZE_EPISODE=0,
            DIAGNOSTICS_BATCH_SIZE=BATCH,
        )
        fill_main_replay_primary(actor)
        actor.freeze_diagnostic_batch()
        out = actor.compute_diagnostics()
        assert isinstance(out, dict) and len(out) > 0


class TestSimNormHelper:
    """simnorm() projects each group onto the simplex."""

    def test_groups_sum_to_one(self):
        x = T.randn(5, 32)
        y = simnorm(x, group_size=8)
        assert y.shape == x.shape
        groups = y.view(5, 4, 8)
        sums = groups.sum(dim=-1)
        assert T.allclose(sums, T.ones_like(sums), atol=1e-5)

    def test_all_nonnegative(self):
        y = simnorm(T.randn(3, 16), group_size=8)
        assert (y >= 0).all()

    def test_rejects_indivisible(self):
        with pytest.raises(ValueError):
            simnorm(T.randn(2, 10), group_size=8)  # 10 % 8 != 0

    def test_handles_arbitrary_leading_dims(self):
        y = simnorm(T.randn(2, 3, 16), group_size=8)
        assert y.shape == (2, 3, 16)
        assert T.allclose(
            y.view(2, 3, 2, 8).sum(dim=-1), T.ones(2, 3, 2), atol=1e-5
        )


class TestSimNormEncoder:
    """GSP_JEPA_SIMNORM makes the encoder output lie on the simplex."""

    def test_encoder_output_on_simplex_when_on(self):
        actor = make_actor(GSP_JEPA_SIMNORM=True)
        assert actor.gsp_encoder_online.simnorm is True
        # Target encoder (EMA deepcopy) must inherit the flag.
        assert actor.gsp_encoder_target.simnorm is True
        x = T.randn(BATCH, GSP_INPUT).to(actor.gsp_encoder_online.device)
        with T.no_grad():
            z = actor.gsp_encoder_online(x)
        assert z.shape == (BATCH, ENC_DIM)
        groups = z.view(BATCH, ENC_DIM // 8, 8)
        assert T.allclose(
            groups.sum(dim=-1), T.ones(BATCH, ENC_DIM // 8).to(z.device), atol=1e-5
        )

    def test_encoder_output_unbounded_when_off(self):
        """Flag OFF: the encoder output is NOT on the simplex (raw linear)."""
        actor = make_actor()  # SimNorm off
        assert actor.gsp_encoder_online.simnorm is False
        x = T.randn(BATCH, GSP_INPUT).to(actor.gsp_encoder_online.device)
        with T.no_grad():
            z = actor.gsp_encoder_online(x)
        groups = z.view(BATCH, ENC_DIM // 8, 8)
        sums = groups.sum(dim=-1)
        assert not T.allclose(
            sums, T.ones_like(sums), atol=1e-3
        ), "raw encoder output should not already be on the simplex"

    def test_simnorm_coupled_learn_step_runs(self):
        actor = make_actor(GSP_JEPA_SIMNORM=True, GSP_JEPA_COUPLE_VALUE=True)
        fill_main_replay(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None and np.isfinite(result["jepa_pred_mse"])

    def test_simnorm_rejects_bad_group_size_at_build(self):
        # ENC_DIM=8 is not divisible by group_size=3.
        with pytest.raises(ValueError):
            make_actor(GSP_JEPA_SIMNORM=True, GSP_JEPA_SIMNORM_GROUP_SIZE=3)


class TestLatentPrimaryAndSimNormCompose:
    """Both flags together: input dim dropped AND latent on simplex."""

    def test_compose_dims_and_learn(self):
        actor = make_actor(
            GSP_ACTOR_LATENT_PRIMARY=True,
            GSP_JEPA_SIMNORM=True,
            GSP_JEPA_COUPLE_VALUE=True,
        )
        assert actor.network_input_size == ENC_DIM
        assert actor.gsp_encoder_online.simnorm is True
        fill_main_replay_primary(actor)
        result = actor.learn_DDQN_jepa_coupled(actor.networks, actor.gsp_networks)
        assert result is not None and np.isfinite(result["total_loss"])


class TestNewFlagsOffByteIdentical:
    """With the two new flags OFF, network dims, replay dims, and encoder
    forward are byte-identical to a plain coupled build — cannot change legacy."""

    def test_network_dims_unchanged_when_flags_off(self):
        base = make_actor(GSP_JEPA_COUPLE_VALUE=True)
        withflags = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_ACTOR_LATENT_PRIMARY=False,
            GSP_JEPA_SIMNORM=False,
        )
        assert base.network_input_size == withflags.network_input_size == ENV_OBS + ENC_DIM
        assert (
            base.networks["q_eval"].fc1.weight.shape
            == withflags.networks["q_eval"].fc1.weight.shape
        )
        assert (
            base.networks["replay"].gsp_obs_size
            == withflags.networks["replay"].gsp_obs_size
        )

    def test_encoder_forward_identical_when_flags_off(self):
        T.manual_seed(7)
        np.random.seed(7)
        a1 = make_actor(GSP_JEPA_COUPLE_VALUE=True)
        T.manual_seed(7)
        np.random.seed(7)
        a2 = make_actor(
            GSP_JEPA_COUPLE_VALUE=True,
            GSP_ACTOR_LATENT_PRIMARY=False,
            GSP_JEPA_SIMNORM=False,
        )
        x = T.randn(BATCH, GSP_INPUT).to(a1.gsp_encoder_online.device)
        with T.no_grad():
            z1 = a1.gsp_encoder_online(x)
            z2 = a2.gsp_encoder_online(x.to(a2.gsp_encoder_online.device))
        assert T.allclose(z1.cpu(), z2.cpu(), atol=0.0)
