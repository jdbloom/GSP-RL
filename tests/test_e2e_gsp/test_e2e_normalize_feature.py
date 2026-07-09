"""Tests for GSP_E2E_NORMALIZE_FEATURE (opt-in unit-variance standardization of
the spliced GSP feature) at the E2E LEARN splice.

The acting-side splice (RL-CT agent.make_agent_state) is covered in the
RL-CollectiveTransport suite; the shared-stats consistency contract is asserted
in both suites. Here we cover the GSP-RL learn path:

  (b) flag OFF -> byte-identical to baseline (gsp_feature_stats is None, no-op).
  (b') flag default is False; config key parsed under GSP_E2E_NORMALIZE_FEATURE.
  (n) with an injected standardizer, the spliced actor slot is standardized
      (matches standardizer.standardize(scaled_pred)), and the head's supervised
      MSE stays on the RAW prediction.
  (d) the learn step UPDATES the shared stats (train-time update policy).

All tests use injected/synthetic data (no experiment artifacts) -> zero-spend.
"""
import copy
import numpy as np
import torch as T

from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.actors.feature_stats import RunningStandardizer

from .test_e2e_learn_step import (
    make_aids,
    make_ddqn_networks,
    make_gsp_networks,
    fill_replay,
    MINIMAL_CONFIG,
    ENV_OBS_SIZE,
    MEM_SIZE,
)


def _setup():
    aids = make_aids()
    networks = make_ddqn_networks()
    gsp_networks = make_gsp_networks()
    fill_replay(networks['replay'], MEM_SIZE)
    return aids, networks, gsp_networks


class TestFlagDefaultAndParsing:
    def test_default_flag_is_false(self):
        aids = NetworkAids(dict(MINIMAL_CONFIG))
        assert aids.gsp_e2e_normalize_feature is False
        assert aids.gsp_feature_stats is None

    def test_config_key_read(self):
        cfg = dict(MINIMAL_CONFIG)
        cfg['GSP_E2E_NORMALIZE_FEATURE'] = True
        aids = NetworkAids(cfg)
        assert aids.gsp_e2e_normalize_feature is True


class TestFlagOffIsIdentical:
    """(b) Flag OFF -> the learn step is byte-identical to baseline (no stats
    object, no standardization touches the spliced slot)."""

    def test_flag_off_gsp_feature_stats_none_and_slot_is_plain_scaled_pred(self):
        aids, networks, gsp_networks = _setup()
        assert aids.gsp_feature_stats is None  # flag off default

        cap = {}
        head_fwd = gsp_networks['actor'].forward
        q_fwd = networks['q_eval'].forward

        def head_spy(x):
            out = head_fwd(x)
            cap['pred'] = out.detach().clone()
            return out

        def q_spy(x):
            if 'aug' not in cap:
                cap['aug'] = x.detach().clone()
            return q_fwd(x)

        gsp_networks['actor'].forward = head_spy
        networks['q_eval'].forward = q_spy
        aids.learn_DDQN_e2e(networks, gsp_networks)

        scale = float(np.degrees(1.0) / 10.0)
        pred = cap['pred'].reshape(-1, 1)
        slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + 1]
        # Flag off => slot is exactly the scaled raw pred (no standardization).
        assert T.allclose(slot, pred * scale, atol=1e-5)

    def test_flag_off_learn_step_bit_identical(self):
        """Two aids that start from identical network+replay state and differ ONLY
        in that one explicitly has GSP_E2E_NORMALIZE_FEATURE set (to False) must
        produce bit-identical q_eval params after one learn step. The replay sample
        draw uses global np.random, so both runs are reseeded to the SAME seed
        immediately before their learn step to isolate the flag as the only
        variable."""
        T.manual_seed(0)
        a1, n1, g1 = _setup()
        cfg = dict(MINIMAL_CONFIG)
        cfg['GSP_E2E_NORMALIZE_FEATURE'] = False
        a2 = make_aids_with_cfg(cfg)
        n2, g2 = clone_networks_from(n1, g1)

        np.random.seed(123)
        a1.learn_DDQN_e2e(n1, g1)
        np.random.seed(123)
        a2.learn_DDQN_e2e(n2, g2)

        assert a1.gsp_feature_stats is None and a2.gsp_feature_stats is None
        for k in n1['q_eval'].state_dict():
            assert T.allclose(
                n1['q_eval'].state_dict()[k], n2['q_eval'].state_dict()[k]
            ), f"q_eval param {k} diverged with flag explicitly off"


class TestFlagOnStandardizesActorSlotOnly:
    """(n) With an injected standardizer, the actor slot is the STANDARDIZED scaled
    pred, while the head's supervised MSE stays on the RAW pred."""

    def test_actor_slot_is_standardized_and_mse_uses_raw(self):
        import gsp_rl.src.actors.learning_aids as la

        aids, networks, gsp_networks = _setup()
        # Inject a pre-warmed standardizer (as Actor.__init__ would create; warmed
        # so it is not the count==0 identity).
        stats = RunningStandardizer(dim=1)
        stats.update(np.random.default_rng(0).normal(1.7, 0.3, size=(512, 1)))
        aids.gsp_e2e_normalize_feature = True
        aids.gsp_feature_stats = stats
        # Snapshot the pre-step stats so we can reproduce the standardization the
        # learn step applied (it standardizes with pre-batch stats, THEN updates).
        pre_mean = stats.mean.copy()
        pre_var = stats.var.copy()

        cap = {}
        head_fwd = gsp_networks['actor'].forward
        q_fwd = networks['q_eval'].forward
        real_mse = la.F.mse_loss

        def head_spy(x):
            out = head_fwd(x)
            cap['pred'] = out.detach().clone()
            return out

        def q_spy(x):
            if 'aug' not in cap:
                cap['aug'] = x.detach().clone()
            return q_fwd(x)

        def mse_spy(pred, target, *a, **kw):
            cap['mse_pred'] = pred.detach().clone()
            return real_mse(pred, target, *a, **kw)

        gsp_networks['actor'].forward = head_spy
        networks['q_eval'].forward = q_spy
        la.F.mse_loss = mse_spy
        try:
            aids.learn_DDQN_e2e(networks, gsp_networks)
        finally:
            la.F.mse_loss = real_mse

        scale = float(np.degrees(1.0) / 10.0)
        pred = cap['pred'].reshape(-1, 1).cpu()
        scaled = pred * scale
        std_arr = np.sqrt(pre_var + stats.eps).astype(np.float32)
        expected_slot = (scaled.numpy() - pre_mean.astype(np.float32)) / std_arr
        slot = cap['aug'][:, ENV_OBS_SIZE:ENV_OBS_SIZE + 1].cpu().numpy()
        np.testing.assert_allclose(slot, expected_slot, rtol=1e-4, atol=1e-5)

        # The slot must NOT be the plain scaled pred anymore (standardization moved it).
        assert not np.allclose(slot, scaled.numpy(), atol=1e-3)

        # The supervised MSE input is the RAW head pred, untouched by standardization.
        np.testing.assert_allclose(
            cap['mse_pred'].reshape(-1, 1).cpu().numpy(), pred.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_learn_step_updates_shared_stats(self):
        """(d) The learn step folds the batch into the shared stats (train-time
        update policy) — count grows by the batch size."""
        aids, networks, gsp_networks = _setup()
        stats = RunningStandardizer(dim=1)
        stats.update(np.zeros((10, 1)))
        aids.gsp_e2e_normalize_feature = True
        aids.gsp_feature_stats = stats
        before = stats.count
        aids.learn_DDQN_e2e(networks, gsp_networks)
        assert stats.count == before + aids.batch_size


def make_aids_with_cfg(cfg):
    aids = NetworkAids(cfg)
    aids.input_size = ENV_OBS_SIZE
    return aids


def clone_networks_from(networks, gsp_networks):
    """Deep-copy the DDQN + GSP networks (incl. replay) so a second learn step runs
    on identical starting state."""
    n2 = copy.deepcopy(networks)
    g2 = copy.deepcopy(gsp_networks)
    return n2, g2
