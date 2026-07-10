"""Integration tests: the actor-GSP-feature reliance diagnostic is emitted by the
E2E learn steps, reflects the standardizer, and does NOT perturb training.

Headline metric for the causal-usage investigation: does the actor's first-layer
weight lean on the spliced GSP prediction over training? These tests assert the
metric is present in both the DDQN and TD3 e2e diagnostics dicts, that the
prenorm/postnorm feature-std pair reflects the NORMALIZE_FEATURE standardizer,
and — critically — that computing the diagnostic leaves the learned parameters
bit-identical (pure read under no_grad).

All synthetic data, no experiment artifacts -> zero-spend.
"""
import copy
import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.feature_stats import RunningStandardizer

from .test_e2e_learn_step import (
    make_aids,
    make_ddqn_networks,
    make_gsp_networks,
    fill_replay,
    ENV_OBS_SIZE,
    MEM_SIZE,
)
from .test_td3_e2e_learn_step import (
    make_aids as make_aids_td3,
    make_td3_networks,
    make_gsp_networks as make_gsp_networks_td3,
    fill_replay as fill_replay_td3,
    ENV_OBS_SIZE as ENV_OBS_SIZE_TD3,
    MEM_SIZE as MEM_SIZE_TD3,
)


WEIGHT_KEYS = {
    'actor_gsp_feature_weight_norm',
    'actor_obs_weight_norm_mean',
    'actor_gsp_weight_ratio',
    'gsp_feature_std_prenorm',
    'gsp_feature_std_postnorm',
}


def _ddqn_setup():
    aids = make_aids()
    networks = make_ddqn_networks()
    gsp_networks = make_gsp_networks()
    fill_replay(networks['replay'], MEM_SIZE)
    return aids, networks, gsp_networks


def _td3_setup():
    aids = make_aids_td3()
    networks = make_td3_networks()
    gsp_networks = make_gsp_networks_td3()
    fill_replay_td3(networks['replay'], MEM_SIZE_TD3)
    return aids, networks, gsp_networks


class TestDDQNDiagnosticsPresent:
    def test_all_weight_keys_present(self):
        aids, networks, gsp_networks = _ddqn_setup()
        result = aids.learn_DDQN_e2e(networks, gsp_networks)
        missing = WEIGHT_KEYS - set(result.keys())
        assert not missing, f"Missing weight-diag keys: {missing}"

    def test_weight_metrics_finite_and_nonneg(self):
        aids, networks, gsp_networks = _ddqn_setup()
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        for k in ('actor_gsp_feature_weight_norm', 'actor_obs_weight_norm_mean',
                  'actor_gsp_weight_ratio', 'gsp_feature_std_prenorm',
                  'gsp_feature_std_postnorm'):
            assert np.isfinite(r[k]), f"{k} not finite: {r[k]}"
            assert r[k] >= 0.0, f"{k} negative: {r[k]}"

    def test_weight_norm_matches_fc1_last_column(self):
        # With K=1 the reported GSP feature weight norm must equal the L2 norm of
        # the last column of the Q-net fc1 weight (post-step).
        aids, networks, gsp_networks = _ddqn_setup()
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        w = networks['q_eval'].fc1.weight.detach()
        expected = float(T.linalg.vector_norm(w[:, ENV_OBS_SIZE]).item())
        assert r['actor_gsp_feature_weight_norm'] == pytest.approx(expected, rel=1e-5)


class TestTD3DiagnosticsPresent:
    def test_all_weight_keys_present(self):
        aids, networks, gsp_networks = _td3_setup()
        result = aids.learn_TD3_e2e(networks, gsp_networks)
        missing = WEIGHT_KEYS - set(result.keys())
        assert not missing, f"Missing weight-diag keys: {missing}"

    def test_weight_norm_matches_actor_fc1_last_column(self):
        aids, networks, gsp_networks = _td3_setup()
        r = aids.learn_TD3_e2e(networks, gsp_networks)
        w = networks['actor'].fc1.weight.detach()
        expected = float(T.linalg.vector_norm(w[:, ENV_OBS_SIZE_TD3]).item())
        assert r['actor_gsp_feature_weight_norm'] == pytest.approx(expected, rel=1e-5)


class TestFeatureStdReflectsStandardizer:
    def test_normalize_off_post_equals_pre(self):
        # No standardizer -> post == pre (identity).
        aids, networks, gsp_networks = _ddqn_setup()
        assert aids.gsp_feature_stats is None
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        assert r['gsp_feature_std_postnorm'] == pytest.approx(r['gsp_feature_std_prenorm'])

    def test_normalize_on_rescales_toward_unit(self):
        # Warm the standardizer so it has non-trivial stats, then confirm the
        # post-standardization feature std differs from pre and lands near unit
        # variance (the whole point of the standardizer).
        aids, networks, gsp_networks = _ddqn_setup()
        stats = RunningStandardizer(dim=1)
        # Warm with a batch whose std is far from 1 so standardization is visible.
        stats.update(np.random.default_rng(0).normal(5.0, 0.02, size=(512, 1)))
        aids.gsp_e2e_normalize_feature = True
        aids.gsp_feature_stats = stats
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        # The raw (pre) feature is a tiny-std scalar; post-standardization divides
        # by the standardizer std (~0.02) so the post std must be strictly larger.
        assert r['gsp_feature_std_postnorm'] != pytest.approx(r['gsp_feature_std_prenorm'])
        assert r['gsp_feature_std_postnorm'] > r['gsp_feature_std_prenorm']


class TestSpliceGain:
    def test_default_gain_is_one_and_noop(self):
        aids, networks, gsp_networks = _ddqn_setup()
        assert aids.gsp_e2e_splice_gain == 1.0
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        # gain 1.0 -> post == pre exactly (normalize off).
        assert r['gsp_feature_std_postnorm'] == pytest.approx(r['gsp_feature_std_prenorm'])

    def test_gain_scales_postnorm_by_constant(self):
        # The postnorm diagnostic reads AFTER the gain — the scale the actor
        # sees. With normalize off, post must equal gain × pre.
        aids, networks, gsp_networks = _ddqn_setup()
        aids.gsp_e2e_splice_gain = 10.0
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        assert r['gsp_feature_std_postnorm'] == pytest.approx(
            10.0 * r['gsp_feature_std_prenorm'], rel=1e-5
        )

    def test_gain_composes_after_standardizer(self):
        # normalize on + gain: gain is the LAST transform, so within one run
        # post == gain × prenorm / stats.std exactly (standardize is an affine
        # (x − m)/s, which scales the std by 1/s; the gain then multiplies it).
        aids, networks, gsp_networks = _ddqn_setup()
        stats = RunningStandardizer(dim=1)
        stats.update(np.random.default_rng(0).normal(5.0, 0.02, size=(512, 1)))
        aids.gsp_e2e_normalize_feature = True
        aids.gsp_feature_stats = stats
        aids.gsp_e2e_splice_gain = 3.0
        std_before_update = float(stats.std[0])
        r = aids.learn_DDQN_e2e(networks, gsp_networks)
        assert r['gsp_feature_std_postnorm'] == pytest.approx(
            3.0 * r['gsp_feature_std_prenorm'] / std_before_update, rel=1e-3
        )

    def test_td3_gain_scales_postnorm(self):
        aids, networks, gsp_networks = _td3_setup()
        aids.gsp_e2e_splice_gain = 10.0
        r = aids.learn_TD3_e2e(networks, gsp_networks)
        assert r['gsp_feature_std_postnorm'] == pytest.approx(
            10.0 * r['gsp_feature_std_prenorm'], rel=1e-5
        )


class TestNoPerturbation:
    """The diagnostic must not change training dynamics: a run WITH the metric
    must be bit-identical in learned params to one WITHOUT it.

    We patch out the diagnostic method to simulate 'without', run the same seeded
    learn step on a fresh identical net, and assert every parameter tensor is
    bit-identical to the 'with' run.
    """

    def _snapshot(self, *modules):
        out = {}
        for i, m in enumerate(modules):
            for k, v in m.state_dict().items():
                out[f"{i}.{k}"] = v.clone()
        return out

    def test_ddqn_metric_does_not_perturb_params(self):
        # WITH the metric
        T.manual_seed(1234)
        np.random.seed(1234)
        aids_a, net_a, gsp_a = _ddqn_setup()
        aids_a.learn_DDQN_e2e(net_a, gsp_a)
        with_snap = self._snapshot(net_a['q_eval'], gsp_a['actor'])

        # WITHOUT the metric: monkeypatch the diagnostic to a no-op returning the
        # keys with NaN so the return-dict shape is unaffected but no weight read
        # happens. Identical seeds + identical construction order.
        T.manual_seed(1234)
        np.random.seed(1234)
        aids_b, net_b, gsp_b = _ddqn_setup()
        nan = float('nan')
        aids_b._actor_gsp_weight_diag = lambda *a, **k: {
            'actor_gsp_feature_weight_norm': nan,
            'actor_obs_weight_norm_mean': nan,
            'actor_gsp_weight_ratio': nan,
        }
        aids_b.learn_DDQN_e2e(net_b, gsp_b)
        without_snap = self._snapshot(net_b['q_eval'], gsp_b['actor'])

        for key in with_snap:
            assert T.equal(with_snap[key], without_snap[key]), (
                f"param {key} differs with vs without the diagnostic — the metric "
                f"perturbed training dynamics"
            )
