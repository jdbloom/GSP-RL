"""GSP_SPLICE_ADVANTAGE_ONLY — advantage-stream dueling splice tests.

Contract under test (opt-in, default off):

(a) Flag OFF (or absent) = byte-identical legacy flat Q-head: no extra
    modules, no extra RNG draws at construction, bit-exact forward outputs
    (same technique as tests/test_actor/test_batched_gsp_head.py's off-path
    golden checks).
(b) Flag ON (DQN/DDQN + GSP splice) = dueling head with the split placed so
    the spliced GSP prediction reaches ONLY the advantage stream:
      * V(s) is computed from the pred-EXCLUDED input — perturbing the pred
        columns leaves V (== mean_a Q, by the mean(A) subtraction) bit-exact
        unchanged while Q's differential component moves;
      * Q = V + A - mean(A) identity holds;
      * the Q-loss gradient still flows into the pred input columns (the E2E
        path into the GSP head stays alive).
(c) Unsupported schemes (DDPG/TD3/RDDPG, SF head, JEPA splice, no GSP slot)
    raise loudly at construction — never silently ignore the flag.

Motivation: the 2026-07-09 Q-probe measured the flat head absorbing the
spliced prediction ~99.8% as common-mode (state-value offset). The dueling
split architecturally forbids that absorption. All synthetic data, zero-spend.
"""
import copy

import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors import Actor
from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.buffers.replay import ReplayBuffer
from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.dqn import DQN

K = 5  # delta_theta_traj horizon — the campaign feature width
OBS = 31
AUG = OBS + K

# --- Mirrors tests/test_e2e_gsp/test_e2e_learn_step.py's minimal fixtures
# (duplicated: tests/ is not a package, so no cross-directory import). ---
ENV_OBS_SIZE = 31
GSP_OBS_SIZE = 6
GSP_OUTPUT_SIZE = 1
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
    aids = NetworkAids(MINIMAL_CONFIG)
    aids.input_size = ENV_OBS_SIZE
    return aids


def make_gsp_networks() -> dict:
    actor = DDPGActorNetwork(
        id=0, lr=LR, input_size=GSP_OBS_SIZE, output_size=GSP_OUTPUT_SIZE,
        fc1_dims=32, fc2_dims=16, min_max_action=1.0, use_linear_output=True,
    )
    return {'actor': actor, 'learning_scheme': 'DDPG', 'learn_step_counter': 0}


def fill_replay(replay: ReplayBuffer, n: int) -> None:
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



def _tt(net, arr, requires_grad=False):
    """Test tensor on the net's device (nets auto-place on cuda/mps/cpu)."""
    t = T.tensor(np.asarray(arr), dtype=T.float32, device=net.device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def _actor_args(**overrides):
    args = {
        'id': 0,
        'config': dict(MINIMAL_CONFIG),
        'network': 'DDQN',
        'input_size': OBS,
        'output_size': 9,
        'min_max_action': 1.0,
        'meta_param_size': 2,
        'gsp': True,
        'gsp_input_size': GSP_OBS_SIZE,
        'gsp_output_size': 1,
    }
    args.update(overrides)
    return args


def _make_actor(seed=0, config_overrides=None, **overrides):
    args = _actor_args(**overrides)
    if config_overrides:
        args['config'] = {**args['config'], **config_overrides}
    T.manual_seed(seed)
    return Actor(**args)


def _traj_config(**extra):
    """Config for the K=5 delta_theta_traj campaign shape."""
    cfg = {
        'GSP_OUTPUT_KIND': 'delta_theta_traj',
        'GSP_PREDICTION_HORIZON': K,
    }
    cfg.update(extra)
    return cfg


def _dueling_ddqn(seed=3, use_layer_norm=False, fc1=32, fc2=32):
    T.manual_seed(seed)
    return DDQN(
        id=0, lr=LR, input_size=AUG, output_size=NUM_ACTIONS,
        fc1_dims=fc1, fc2_dims=fc2, use_layer_norm=use_layer_norm,
        advantage_only_pred=(OBS, K),
    )


class TestFlagParsing:
    def test_default_is_off(self):
        actor = _make_actor()
        assert actor.gsp_splice_advantage_only is False
        assert actor.gsp_splice_advantage_engaged is False

    def test_flag_on_engages_for_ddqn_gsp(self):
        actor = _make_actor(
            config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': True})
        assert actor.gsp_splice_advantage_only is True
        assert actor.gsp_splice_advantage_engaged is True


class TestFlagOffByteIdentical:
    """(a) Off-path golden: absent and explicit-False builds are bit-exact."""

    def test_state_dicts_bit_exact_absent_vs_false(self):
        a_absent = _make_actor(seed=7)
        a_false = _make_actor(
            seed=7, config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': False})
        sd_a = a_absent.networks['q_eval'].state_dict()
        sd_f = a_false.networks['q_eval'].state_dict()
        assert set(sd_a.keys()) == set(sd_f.keys())
        for k in sd_a:
            assert T.equal(sd_a[k], sd_f[k]), f'{k} differs'

    def test_no_value_stream_modules_when_off(self):
        actor = _make_actor(seed=7)
        assert actor.networks['q_eval'].advantage_only_pred is None
        assert not hasattr(actor.networks['q_eval'], 'v_fc1')

    def test_forward_bit_exact_absent_vs_false(self):
        a_absent = _make_actor(seed=11)
        a_false = _make_actor(
            seed=11, config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': False})
        x = _tt(a_absent.networks['q_eval'],
                np.random.default_rng(5).standard_normal((16, OBS + 1)))
        with T.no_grad():
            qa = a_absent.networks['q_eval'](x)
            qf = a_false.networks['q_eval'](x)
        assert T.equal(qa, qf)

    def test_construction_consumes_identical_rng_stream(self):
        """No extra RNG draws when off: the next torch draw after construction
        matches between absent and explicit-False builds."""
        _make_actor(seed=13)
        after_absent = T.rand(8)
        _make_actor(
            seed=13, config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': False})
        after_false = T.rand(8)
        assert T.equal(after_absent, after_false)


class TestDuelingNetworkOn:
    """(b) ON-path unit tests on the DDQN network itself."""

    def test_value_stream_modules_built(self):
        net = _dueling_ddqn()
        assert net.advantage_only_pred == (OBS, K)
        assert net.v_fc1.in_features == OBS
        assert net.v_fc3.out_features == 1

    @pytest.mark.parametrize('use_layer_norm', [False, True])
    def test_v_invariant_to_pred_perturbation_advantage_moves(
            self, use_layer_norm):
        """Perturb ONLY the pred columns: V (== Q.mean(-1)) must be bit-exact
        unchanged; the differential component (Q - Q.mean) must move."""
        net = _dueling_ddqn(use_layer_norm=use_layer_norm)
        rng = np.random.default_rng(17)
        x = _tt(net, rng.standard_normal((32, AUG)))
        x_pert = x.clone()
        x_pert[:, OBS:] += _tt(net, rng.standard_normal((32, K)))
        with T.no_grad():
            q, q_pert = net(x), net(x_pert)
            v, v_pert = net.value_stream(x), net.value_stream(x_pert)
        # Direct stream check AND the identity-derived check (V == mean_a Q).
        assert T.equal(v, v_pert)
        assert T.allclose(q.mean(dim=-1), q_pert.mean(dim=-1), atol=1e-6)
        diff = q - q.mean(dim=-1, keepdim=True)
        diff_pert = q_pert - q_pert.mean(dim=-1, keepdim=True)
        assert not T.allclose(diff, diff_pert)

    def test_obs_perturbation_moves_v(self):
        """Guard against a dead value stream: perturbing the OBS columns must
        move V."""
        net = _dueling_ddqn()
        rng = np.random.default_rng(19)
        x = _tt(net, rng.standard_normal((32, AUG)))
        x_pert = x.clone()
        x_pert[:, :OBS] += _tt(net, rng.standard_normal((32, OBS)))
        with T.no_grad():
            v, v_pert = net.value_stream(x), net.value_stream(x_pert)
        assert not T.allclose(v, v_pert)

    def test_q_equals_v_plus_advantage_minus_mean(self):
        """Q = V + A - mean(A), where A is the trunk (fc1/fc2/fc3) output."""
        net = _dueling_ddqn()
        rng = np.random.default_rng(23)
        x = _tt(net, rng.standard_normal((8, AUG)))
        with T.no_grad():
            q = net(x)
            # Recompute the advantage trunk manually (legacy forward ops).
            h = T.relu(net.fc1(x))
            h = T.relu(net.fc2(h))
            a = net.fc3(h)
            v = net.value_stream(x)
            expected = v + a - a.mean(dim=-1, keepdim=True)
        assert T.allclose(q, expected, atol=1e-6)

    def test_one_dim_input_matches_batched_row(self):
        """choose_action feeds a 1-D (AUG,) tensor; the dim=-1 slicing must
        give the same Q as the batched path."""
        net = _dueling_ddqn()
        rng = np.random.default_rng(27)
        x = _tt(net, rng.standard_normal(AUG))
        with T.no_grad():
            q_single = net(x)
            q_batched = net(x.unsqueeze(0)).squeeze(0)
        assert T.allclose(q_single, q_batched, atol=1e-6)

    def test_gradient_flows_from_q_loss_into_pred_columns(self):
        """The E2E path: a Q loss must produce nonzero grad at the pred input
        columns (through the advantage stream) and at the obs columns."""
        net = _dueling_ddqn()
        rng = np.random.default_rng(29)
        x = _tt(net, rng.standard_normal((16, AUG)), requires_grad=True)
        q = net(x)
        loss = (q ** 2).sum()
        loss.backward()
        pred_grad = x.grad[:, OBS:].abs().sum()
        obs_grad = x.grad[:, :OBS].abs().sum()
        assert float(pred_grad) > 0.0
        assert float(obs_grad) > 0.0

    def test_value_stream_params_get_no_grad_from_pred_only_path(self):
        """Grad wiring: d(Q_differential)/d(v_fc*) == 0 — the value stream
        never learns from the differential component the pred drives."""
        net = _dueling_ddqn()
        rng = np.random.default_rng(31)
        x = _tt(net, rng.standard_normal((16, AUG)))
        q = net(x)
        diff = q - q.mean(dim=-1, keepdim=True)
        (diff ** 2).sum().backward()
        for name in ('v_fc1', 'v_fc2', 'v_fc3'):
            g = getattr(net, name).weight.grad
            assert g is None or float(g.abs().max()) < 1e-5, (
                f'{name} received gradient from the differential (advantage) '
                'component — V/A separation is broken'
            )

    def test_bad_span_raises(self):
        with pytest.raises(ValueError):
            DDQN(id=0, lr=LR, input_size=AUG, output_size=NUM_ACTIONS,
                 advantage_only_pred=(OBS, K + 1))
        with pytest.raises(ValueError):
            DDQN(id=0, lr=LR, input_size=K, output_size=NUM_ACTIONS,
                 advantage_only_pred=(0, K))

    def test_dqn_class_mirrors_ddqn(self):
        T.manual_seed(3)
        net = DQN(id=0, lr=LR, input_size=AUG, output_size=NUM_ACTIONS,
                  fc1_dims=32, fc2_dims=32, advantage_only_pred=(OBS, K))
        rng = np.random.default_rng(33)
        x = _tt(net, rng.standard_normal((8, AUG)))
        x_pert = x.clone()
        x_pert[:, OBS:] += 1.0
        with T.no_grad():
            v, v_pert = net.value_stream(x), net.value_stream(x_pert)
            q, q_pert = net(x), net(x_pert)
        assert T.equal(v, v_pert)
        assert not T.allclose(q, q_pert)

    def test_value_stream_raises_on_flat_head(self):
        T.manual_seed(3)
        net = DDQN(id=0, lr=LR, input_size=AUG, output_size=NUM_ACTIONS)
        with pytest.raises(RuntimeError):
            net.value_stream(T.zeros(AUG, device=net.device))


class TestActorGateEngaged:
    """Actor-level wiring: the engaged build produces dueling q_eval/q_next
    with the campaign span (input_size, K)."""

    def test_ddqn_traj_build_has_dueling_pair(self):
        actor = _make_actor(
            config_overrides=_traj_config(GSP_SPLICE_ADVANTAGE_ONLY=True))
        assert actor.gsp_splice_advantage_engaged is True
        for net_key in ('q_eval', 'q_next'):
            net = actor.networks[net_key]
            assert net.advantage_only_pred == (OBS, K)
            assert net.v_fc1.in_features == OBS
        assert actor.network_input_size == AUG

    def test_dqn_scheme_also_supported(self):
        actor = _make_actor(
            network='DQN',
            config_overrides=_traj_config(GSP_SPLICE_ADVANTAGE_ONLY=True))
        assert actor.gsp_splice_advantage_engaged is True
        assert actor.networks['q_eval'].advantage_only_pred == (OBS, K)

    def test_optimizer_covers_value_stream_params(self):
        actor = _make_actor(
            config_overrides=_traj_config(GSP_SPLICE_ADVANTAGE_ONLY=True))
        net = actor.networks['q_eval']
        opt_params = {id(p) for group in net.optimizer.param_groups
                      for p in group['params']}
        for name in ('v_fc1', 'v_fc2', 'v_fc3'):
            for p in getattr(net, name).parameters():
                assert id(p) in opt_params, f'{name} missing from optimizer'

    def test_checkpoint_roundtrip_preserves_value_stream(self, tmp_path):
        actor = _make_actor(
            seed=5, config_overrides=_traj_config(GSP_SPLICE_ADVANTAGE_ONLY=True))
        net = actor.networks['q_eval']
        net.save_checkpoint(str(tmp_path / 'ck'))
        other = _make_actor(
            seed=6, config_overrides=_traj_config(GSP_SPLICE_ADVANTAGE_ONLY=True))
        other_net = other.networks['q_eval']
        other_net.load_checkpoint(str(tmp_path / 'ck'))
        for k, v in net.state_dict().items():
            assert T.equal(v, other_net.state_dict()[k]), f'{k} not restored'


class TestUnsupportedSchemesRaiseLoudly:
    """(c) Never silently ignore the flag."""

    @pytest.mark.parametrize('scheme', ['DDPG', 'TD3'])
    def test_continuous_schemes_raise(self, scheme):
        with pytest.raises(ValueError, match='GSP_SPLICE_ADVANTAGE_ONLY'):
            _make_actor(
                network=scheme,
                config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': True})

    def test_no_gsp_slot_raises(self):
        with pytest.raises(ValueError, match='no spliced prediction slot'):
            _make_actor(
                gsp=False,
                config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': True})

    def test_sf_head_raises(self):
        with pytest.raises(ValueError, match='successor-features'):
            _make_actor(
                config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': True,
                                  'GSP_SF_ENABLED': True})

    def test_jepa_splice_raises(self):
        with pytest.raises(ValueError, match='JEPA'):
            _make_actor(
                config_overrides={'GSP_SPLICE_ADVANTAGE_ONLY': True,
                                  'GSP_JEPA_ENABLED': True})


class TestE2ELearnStepWithDueling:
    """learn_DDQN_e2e on a dueling q_eval/q_next pair: the splice stays
    differentiable — the actor TD loss reaches the GSP head (STOP_GRAD off),
    and both optimizers still step."""

    @staticmethod
    def _dueling_networks():
        T.manual_seed(41)
        kwargs = dict(id=0, lr=LR, input_size=AUGMENTED_OBS_SIZE,
                      output_size=NUM_ACTIONS, fc1_dims=32, fc2_dims=32,
                      advantage_only_pred=(ENV_OBS_SIZE, 1))
        networks = {
            'q_eval': DDQN(**kwargs),
            'q_next': DDQN(**kwargs),
            'replay': ReplayBuffer(
                max_size=MEM_SIZE, num_observations=AUGMENTED_OBS_SIZE,
                num_actions=1, action_type='Discrete',
                gsp_obs_size=GSP_OBS_SIZE),
            'learning_scheme': 'DDQN',
            'learn_step_counter': 0,
        }
        fill_replay(networks['replay'], MEM_SIZE)
        return networks

    def test_learn_step_runs_and_grad_reaches_gsp_head(self):
        aids = make_aids()
        aids.gsp_e2e_lambda = 1.0
        networks = self._dueling_networks()
        gsp_networks = make_gsp_networks()
        before = copy.deepcopy(gsp_networks['actor'].state_dict())

        result = aids.learn_DDQN_e2e(networks, gsp_networks)

        assert np.isfinite(result['total_loss'])
        assert result['gsp_grad_norm'] > 0.0
        # The head actually moved (E2E gradient path alive through the
        # advantage stream + supervised MSE).
        changed = any(
            not T.equal(before[k], gsp_networks['actor'].state_dict()[k])
            for k in before)
        assert changed

    def test_learn_step_updates_value_stream(self):
        aids = make_aids()
        aids.gsp_e2e_lambda = 1.0
        networks = self._dueling_networks()
        gsp_networks = make_gsp_networks()
        v_before = copy.deepcopy(
            networks['q_eval'].v_fc1.weight.detach())

        aids.learn_DDQN_e2e(networks, gsp_networks)

        assert not T.equal(v_before, networks['q_eval'].v_fc1.weight.detach())
