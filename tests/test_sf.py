"""Tests for the Successor-Features value head (GSP_SF_ENABLED).

Pre-reg: docs/research/2026-07-06-successor-features-escalation-prereg.md

Covers:
  1. SF head output shape: psi is (batch, n_actions, d_phi); forward Q is
     (batch, n_actions).
  2. Q = psi . w reduces correctly (matches an explicit einsum).
  3. Default-off byte-identical: unset GSP_SF_ENABLED builds the legacy DDQN pair
     and stores no phi column.
  4. A tiny learn step runs without NaN and updates both psi and w.
  5. Raw obs is KEPT as the net input (input_size unchanged; not routed through a
     latent) — the latent-primary regression guard.
  6. The causal-ablation handle: psi_ablation='zero' -> Q == 0 by construction.
"""
import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor
from gsp_rl.src.networks import DDQN, DDQN_SF


ENV_OBS = 5
NUM_ACTIONS = 4
D_PHI = 3
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
    }
    cfg.update(overrides)
    return cfg


def make_actor(**cfg_overrides):
    cfg = base_config(**cfg_overrides)
    return Actor(
        id=0,
        config=cfg,
        network="DDQN",
        input_size=ENV_OBS,
        output_size=NUM_ACTIONS,
        min_max_action=1,
        meta_param_size=0,
        gsp=False,
        gsp_input_size=6,
        gsp_output_size=1,
    )


def fill_sf_replay(actor, n=BATCH * 3, phi_dim=D_PHI):
    """Store n transitions with a per-step cumulant phi on the main replay."""
    size = actor.network_input_size
    for _ in range(n):
        s = np.random.randn(size).astype(np.float32)
        s_ = np.random.randn(size).astype(np.float32)
        a = np.random.randint(0, NUM_ACTIONS)
        r = float(np.random.randn())
        phi = np.random.randn(phi_dim).astype(np.float32)
        actor.store_agent_transition(s, a, r, s_, False, phi=phi)


# --------------------------------------------------------------------------- #
# 1 + 2: head shape and Q = psi . w reduction
# --------------------------------------------------------------------------- #
class TestHeadShapeAndReduction:
    def test_psi_output_shape(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        psi = net.psi(x)
        assert tuple(psi.shape) == (BATCH, NUM_ACTIONS, D_PHI)

    def test_forward_q_shape(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        q = net.forward(x)
        assert tuple(q.shape) == (BATCH, NUM_ACTIONS)

    def test_q_equals_psi_dot_w(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        psi = net.psi(x)
        q = net.forward(x)
        q_manual = T.einsum('bad,d->ba', psi, net.w)
        assert T.allclose(q, q_manual, atol=1e-6)

    def test_w_is_learnable_parameter(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        assert net.w.requires_grad
        assert tuple(net.w.shape) == (D_PHI,)


# --------------------------------------------------------------------------- #
# 3: default-off byte-identical build
# --------------------------------------------------------------------------- #
class TestDefaultOff:
    def test_default_builds_plain_ddqn(self):
        actor = make_actor()  # GSP_SF_ENABLED unset
        assert isinstance(actor.networks['q_eval'], DDQN)
        assert not isinstance(actor.networks['q_eval'], DDQN_SF)
        # No phi column allocated.
        assert actor.networks['replay'].phi_size == 0
        assert not hasattr(actor.networks['replay'], 'phi_memory')

    def test_default_learn_uses_scalar_ddqn_path(self):
        """With SF off, learn() must run the legacy learn_DDQN scalar path and
        never touch the SF branch."""
        actor = make_actor()
        size = actor.network_input_size
        for _ in range(BATCH * 3):
            actor.store_agent_transition(
                np.random.randn(size).astype(np.float32),
                np.random.randint(0, NUM_ACTIONS),
                0.5,
                np.random.randn(size).astype(np.float32),
                False,
            )
        loss = actor.learn()
        assert loss is not None
        assert actor.last_e2e_diagnostics is None

    def test_sf_enabled_builds_sf_pair_with_phi_column(self):
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI)
        assert isinstance(actor.networks['q_eval'], DDQN_SF)
        assert isinstance(actor.networks['q_next'], DDQN_SF)
        assert actor.networks['replay'].phi_size == D_PHI


# --------------------------------------------------------------------------- #
# 4: learn step runs without NaN and updates psi + w
# --------------------------------------------------------------------------- #
class TestLearnStep:
    def test_learn_step_no_nan_and_returns_diagnostics(self):
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI)
        fill_sf_replay(actor)
        result = actor.learn_DDQN_sf(actor.networks)
        assert result is not None
        for k in ('sf_psi_loss', 'sf_w_loss', 'sf_psi_norm', 'sf_w_norm',
                  'sf_q_mean', 'sf_q_abs_max', 'w'):
            assert k in result
        assert np.isfinite(result['sf_psi_loss'])
        assert np.isfinite(result['sf_w_loss'])
        assert len(result['w']) == D_PHI

    def test_learn_updates_psi_trunk_and_w(self):
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI)
        fill_sf_replay(actor)
        net = actor.networks['q_eval']
        psi_before = net.psi_head.weight.detach().clone()
        w_before = net.w.detach().clone()
        actor.learn_DDQN_sf(actor.networks)
        assert not T.allclose(psi_before, net.psi_head.weight.detach()), \
            "psi trunk did not update"
        assert not T.allclose(w_before, net.w.detach()), "w did not update"

    def test_learn_via_dispatch_returns_scalar(self):
        """learn() routes DDQN+SF to learn_DDQN_sf and returns a scalar loss."""
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI)
        fill_sf_replay(actor)
        loss = actor.learn()
        assert loss is not None and np.isfinite(loss)
        assert actor.last_e2e_diagnostics is not None
        assert 'sf_psi_norm' in actor.last_e2e_diagnostics

    def test_phi_dim_1_fallback(self):
        """d_phi=1 fallback (phi = [scalar reward]) trains without NaN."""
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=1)
        fill_sf_replay(actor, phi_dim=1)
        result = actor.learn_DDQN_sf(actor.networks)
        assert np.isfinite(result['sf_psi_loss'])

    def test_reward_to_go_w_target(self):
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI,
                           GSP_SF_W_TARGET='reward_to_go')
        fill_sf_replay(actor)
        result = actor.learn_DDQN_sf(actor.networks)
        assert np.isfinite(result['sf_w_loss'])

    def test_stabilizers_bound_value_scale(self):
        """With the gate-task critic stabilizers on, the Q scale stays bounded
        over many learn steps (no divergence)."""
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI,
                           CRITIC_LOSS='huber', GRAD_CLIP_NORM=1.0,
                           Q_TARGET_CLIP=10.0, REWARD_SCALE=0.1)
        fill_sf_replay(actor, n=BATCH * 4)
        last = None
        for _ in range(50):
            last = actor.learn_DDQN_sf(actor.networks)
        assert np.isfinite(last['sf_q_abs_max'])
        assert last['sf_q_abs_max'] < 1e3, "value scale diverged under stabilizers"


# --------------------------------------------------------------------------- #
# 5: raw obs kept
# --------------------------------------------------------------------------- #
class TestRawObsKept:
    def test_net_input_size_is_raw_obs(self):
        """The SF net's fc1 input must equal the raw observation size — the raw
        obs is NOT dropped/routed through a latent (latent-primary's mistake)."""
        actor = make_actor(GSP_SF_ENABLED=True, GSP_SF_PHI_DIM=D_PHI)
        net = actor.networks['q_eval']
        assert net.fc1.in_features == actor.network_input_size == ENV_OBS


# --------------------------------------------------------------------------- #
# 6: causal-ablation handle
# --------------------------------------------------------------------------- #
class TestCausalAblation:
    def test_zero_psi_zeroes_q(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        q_before = net.forward(x)
        assert q_before.abs().sum() > 0
        net.psi_ablation = 'zero'
        q_ablated = net.forward(x)
        assert T.allclose(q_ablated, T.zeros_like(q_ablated)), \
            "zeroing psi must zero Q by construction"

    def test_freeze_mean_removes_state_variation(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        net.psi_ablation = 'freeze_mean'
        psi = net.psi(x)
        # Every batch row equals the mean row -> no per-sample variation.
        assert T.allclose(psi, psi[0:1].expand_as(psi), atol=1e-6)


# --------------------------------------------------------------------------- #
# misc: layer-norm variant, penultimate, checkpoint round-trip
# --------------------------------------------------------------------------- #
class TestNetworkMisc:
    def test_layer_norm_variant_forward(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS, output_size=NUM_ACTIONS,
                      d_phi=D_PHI, use_layer_norm=True, critic_loss='huber')
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        assert tuple(net.forward(x).shape) == (BATCH, NUM_ACTIONS)
        assert tuple(net.penultimate(x).shape) == (BATCH, 128)

    def test_penultimate_shape(self):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS, output_size=NUM_ACTIONS,
                      d_phi=D_PHI, fc2_dims=64)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        assert tuple(net.penultimate(x).shape) == (BATCH, 64)

    def test_checkpoint_round_trip(self, tmp_path):
        net = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                      output_size=NUM_ACTIONS, d_phi=D_PHI, name='SF')
        with T.no_grad():
            net.w.copy_(T.arange(D_PHI, dtype=T.float32))
        path = str(tmp_path / 'ckpt')
        net.save_checkpoint(path)
        net2 = DDQN_SF(id=0, lr=1e-3, input_size=ENV_OBS,
                       output_size=NUM_ACTIONS, d_phi=D_PHI, name='SF')
        net2.load_checkpoint(path)
        assert T.allclose(net.w, net2.w)
        x = T.randn(BATCH, ENV_OBS, device=net.device)
        assert T.allclose(net.forward(x), net2.forward(x), atol=1e-6)
