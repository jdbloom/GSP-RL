"""Unit tests for actor_gsp_feature_weight_diag.

This is the headline causal-usage diagnostic for the E2E GSP investigation:
it measures how strongly the actor's first linear layer weights the spliced
GSP prediction columns relative to a typical obs column.

Coverage:
- weight-norm extraction picks the correct (last-K) GSP columns for a known net shape
- ratio ~0 when the GSP columns are ~0; large when they dominate
- pure read: no autograd edge, no mutation of the input weight (no-perturbation)
- numpy and torch inputs agree
"""
import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.feature_stats import actor_gsp_feature_weight_diag


class TestColumnSelection:
    """The diagnostic must read exactly the LAST K columns as the GSP feature."""

    def test_picks_last_k_columns_known_shape(self):
        # hidden=4, n_obs=3, k=2 -> W is (4, 5); last 2 cols are the GSP feature.
        n_obs, k, hidden = 3, 2, 4
        W = T.zeros(hidden, n_obs + k)
        # obs columns: unit L2 norm each (single 1.0 entry per column)
        for j in range(n_obs):
            W[0, j] = 1.0
        # gsp columns: known norms -> col n_obs has L2 norm 3, col n_obs+1 has L2 norm 4
        W[0, n_obs] = 3.0
        W[0, n_obs + 1] = 4.0
        out = actor_gsp_feature_weight_diag(W, n_obs, k)
        # Frobenius over gsp cols = sqrt(3^2 + 4^2) = 5
        assert out["actor_gsp_feature_weight_norm"] == pytest.approx(5.0)
        # obs per-column norm mean = 1.0
        assert out["actor_obs_weight_norm_mean"] == pytest.approx(1.0)
        # ratio = mean gsp col norm / obs col mean = mean(3,4)/1 = 3.5
        assert out["actor_gsp_weight_ratio"] == pytest.approx(3.5)

    def test_k1_ratio_is_col_over_col(self):
        # For K=1 the ratio must equal gsp_col_norm / obs_col_norm exactly.
        n_obs, k, hidden = 5, 1, 3
        W = T.zeros(hidden, n_obs + k)
        for j in range(n_obs):
            W[0, j] = 2.0  # each obs col has L2 norm 2.0
        W[0, n_obs] = 6.0  # gsp col L2 norm 6.0
        out = actor_gsp_feature_weight_diag(W, n_obs, k)
        assert out["actor_gsp_feature_weight_norm"] == pytest.approx(6.0)
        assert out["actor_obs_weight_norm_mean"] == pytest.approx(2.0)
        assert out["actor_gsp_weight_ratio"] == pytest.approx(3.0)


class TestRatioMagnitude:
    """Ratio ~0 when GSP columns vanish; large when they dominate."""

    def test_ratio_near_zero_when_gsp_columns_zero(self):
        n_obs, k, hidden = 8, 1, 6
        rng = np.random.default_rng(0)
        W = rng.standard_normal((hidden, n_obs + k)).astype(np.float32)
        W[:, n_obs:] = 0.0  # actor ignores the prediction
        out = actor_gsp_feature_weight_diag(T.tensor(W), n_obs, k)
        assert out["actor_gsp_feature_weight_norm"] == pytest.approx(0.0)
        assert out["actor_gsp_weight_ratio"] == pytest.approx(0.0)

    def test_ratio_large_when_gsp_columns_dominate(self):
        n_obs, k, hidden = 8, 1, 6
        rng = np.random.default_rng(1)
        W = (0.01 * rng.standard_normal((hidden, n_obs + k))).astype(np.float32)
        W[:, n_obs:] = 100.0  # actor leans hard on the prediction
        out = actor_gsp_feature_weight_diag(T.tensor(W), n_obs, k)
        assert out["actor_gsp_weight_ratio"] > 50.0


class TestPurity:
    """The diagnostic must not perturb the weight or create an autograd edge."""

    def test_no_mutation_of_input_weight(self):
        n_obs, k, hidden = 4, 2, 5
        W = T.randn(hidden, n_obs + k)
        before = W.clone()
        actor_gsp_feature_weight_diag(W, n_obs, k)
        assert T.equal(W, before), "diagnostic mutated the input weight"

    def test_no_autograd_edge_from_grad_weight(self):
        n_obs, k, hidden = 4, 1, 3
        W = T.randn(hidden, n_obs + k, requires_grad=True)
        out = actor_gsp_feature_weight_diag(W, n_obs, k)
        # returned values are plain python floats, not graph-carrying tensors
        for v in out.values():
            assert isinstance(v, float)
        # and reading the diagnostic must not have populated a grad on W
        assert W.grad is None


class TestNumpyTorchAgreement:
    def test_numpy_matches_torch(self):
        n_obs, k, hidden = 7, 3, 9
        rng = np.random.default_rng(7)
        W = rng.standard_normal((hidden, n_obs + k)).astype(np.float32)
        out_np = actor_gsp_feature_weight_diag(W, n_obs, k)
        out_t = actor_gsp_feature_weight_diag(T.tensor(W), n_obs, k)
        for key in out_np:
            assert out_np[key] == pytest.approx(out_t[key], rel=1e-5)


class TestShapeValidation:
    def test_wrong_width_raises(self):
        W = T.randn(4, 6)
        with pytest.raises(ValueError):
            actor_gsp_feature_weight_diag(W, n_obs=3, k=1)  # 3+1 != 6

    def test_bad_k_raises(self):
        W = T.randn(4, 5)
        with pytest.raises(ValueError):
            actor_gsp_feature_weight_diag(W, n_obs=5, k=0)
