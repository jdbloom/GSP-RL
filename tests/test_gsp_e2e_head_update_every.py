"""Tests for GSP_E2E_HEAD_UPDATE_EVERY config knob and _head_should_update helper."""
import pytest
from gsp_rl.src.actors.learning_aids import _head_should_update


class TestHeadShouldUpdate:
    """Unit tests for the pure _head_should_update helper."""

    def test_every_1_always_true(self):
        """update_every=1 → head updates every step."""
        for counter in range(100):
            assert _head_should_update(counter, 1) is True

    def test_every_10_only_at_multiples(self):
        """update_every=10 → True only when counter % 10 == 0."""
        for counter in range(100):
            expected = (counter % 10 == 0)
            assert _head_should_update(counter, 10) == expected

    def test_every_0_treated_as_always(self):
        """update_every=0 → treated as <=1, always True."""
        for counter in range(50):
            assert _head_should_update(counter, 0) is True

    def test_every_negative_treated_as_always(self):
        """update_every=-5 → treated as <=1, always True."""
        for counter in range(50):
            assert _head_should_update(counter, -5) is True

    def test_counter_zero(self):
        """counter=0 always True regardless of update_every."""
        for every in [1, 2, 5, 10, 100]:
            assert _head_should_update(0, every) is True


class TestHyperparametersDefault:
    """Verify the default construction yields gsp_e2e_head_update_every == 1."""

    def test_default_is_1(self):
        """When GSP_E2E_HEAD_UPDATE_EVERY is not in config, default to 1."""
        from gsp_rl.src.actors.learning_aids import Hyperparameters
        # Minimal config with required keys for Hyperparameters.__init__
        config = {
            'GAMMA': 0.99,
            'TAU': 0.005,
            'ALPHA': 0.001,
            'BETA': 0.001,
            'LR': 0.0001,
            'EPSILON': 1.0,
            'EPS_MIN': 0.01,
            'EPS_DEC': 1e-5,
            'GSP_LEARNING_FREQUENCY': 1000,
            'GSP_BATCH_SIZE': 64,
            'BATCH_SIZE': 64,
            'MEM_SIZE': 100000,
            'REPLACE_TARGET_COUNTER': 1000,
            'NOISE': 0.1,
            'UPDATE_ACTOR_ITER': 2,
            'WARMUP': 1000,
        }
        hp = Hyperparameters(config)
        assert hp.gsp_e2e_head_update_every == 1

    def test_explicit_value(self):
        """Explicit GSP_E2E_HEAD_UPDATE_EVERY is read correctly."""
        from gsp_rl.src.actors.learning_aids import Hyperparameters
        config = {
            'GAMMA': 0.99,
            'TAU': 0.005,
            'ALPHA': 0.001,
            'BETA': 0.001,
            'LR': 0.0001,
            'EPSILON': 1.0,
            'EPS_MIN': 0.01,
            'EPS_DEC': 1e-5,
            'GSP_LEARNING_FREQUENCY': 1000,
            'GSP_E2E_HEAD_UPDATE_EVERY': 10,
            'GSP_BATCH_SIZE': 64,
            'BATCH_SIZE': 64,
            'MEM_SIZE': 100000,
            'REPLACE_TARGET_COUNTER': 1000,
            'NOISE': 0.1,
            'UPDATE_ACTOR_ITER': 2,
            'WARMUP': 1000,
        }
        hp = Hyperparameters(config)
        assert hp.gsp_e2e_head_update_every == 10

    def test_none_coerces_to_1(self):
        """YAML null (None) coerces to 1 — fail-loud, never or-default."""
        from gsp_rl.src.actors.learning_aids import Hyperparameters
        config = {
            'GAMMA': 0.99,
            'TAU': 0.005,
            'ALPHA': 0.001,
            'BETA': 0.001,
            'LR': 0.0001,
            'EPSILON': 1.0,
            'EPS_MIN': 0.01,
            'EPS_DEC': 1e-5,
            'GSP_LEARNING_FREQUENCY': 1000,
            'GSP_E2E_HEAD_UPDATE_EVERY': None,
            'GSP_BATCH_SIZE': 64,
            'BATCH_SIZE': 64,
            'MEM_SIZE': 100000,
            'REPLACE_TARGET_COUNTER': 1000,
            'NOISE': 0.1,
            'UPDATE_ACTOR_ITER': 2,
            'WARMUP': 1000,
        }
        hp = Hyperparameters(config)
        assert hp.gsp_e2e_head_update_every == 1