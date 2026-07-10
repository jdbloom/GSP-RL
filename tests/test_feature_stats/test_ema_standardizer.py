"""Unit tests for the EMA mode of RunningStandardizer
(GSP_E2E_NORMALIZE_EMA_HALFLIFE).

Motivating failure (2026-07-09, live cells): at high GSP_E2E_LAMBDA (40000) the
head's early outputs are large/noisy and permanently inflate the all-history
Welford running std — post-norm feature std reached only 0.17-0.39 instead of
~1.0, silently re-shrinking the feature the standardizer exists to normalize.
The EMA mode forgets the inflated early phase with a configurable half-life in
UPDATE counts, so the stats converge to the RECENT feature distribution.

All tests use synthetic/injected numbers only (no experiment data) — zero-spend
and deterministic.
"""

import numpy as np
import pytest
import torch as T

from gsp_rl.src.actors.feature_stats import RunningStandardizer


# ---------------------------------------------------------------------------
# The inflated-early-phase scenario (the headline behavior)
# ---------------------------------------------------------------------------

HALFLIFE = 50.0
N_LARGE_UPDATES = 100    # early phase: large/noisy head outputs
N_SMALL_UPDATES = 1000   # settled phase: the real feature distribution
BATCH = 32
LARGE_STD = 10.0
SMALL_MEAN, SMALL_STD = 0.3, 0.5


def _feed_shifted(std: RunningStandardizer):
    """Feed the two-regime stream (large-variance early phase, then a long
    small-variance settled phase) and return the rng so callers can draw a
    fresh settled-regime batch from the SAME stream position."""
    rng = np.random.default_rng(7)
    for _ in range(N_LARGE_UPDATES):
        std.update(rng.normal(0.0, LARGE_STD, size=(BATCH, 1)))
    for _ in range(N_SMALL_UPDATES):
        std.update(rng.normal(SMALL_MEAN, SMALL_STD, size=(BATCH, 1)))
    return rng


def test_ema_recovers_recent_regime_after_shift():
    """After the distribution shift, the EMA stats converge to the recent
    (settled) regime: a fresh settled-regime batch standardizes to ~zero-mean,
    ~unit-std — the early inflated phase has been forgotten."""
    ema = RunningStandardizer(dim=1, ema_halflife=HALFLIFE)
    rng = _feed_shifted(ema)
    fresh = rng.normal(SMALL_MEAN, SMALL_STD, size=(4096, 1))
    out = ema.standardize(fresh)
    assert out.std() == pytest.approx(1.0, abs=0.1)
    assert abs(out.mean()) < 0.1


def test_legacy_welford_stays_inflated_after_shift():
    """The same stream through the legacy all-history Welford stays inflated:
    the early large-variance phase permanently dominates the running std, so
    the settled regime standardizes to well under unit std (the measured
    0.17-0.39 failure)."""
    legacy = RunningStandardizer(dim=1)
    rng = _feed_shifted(legacy)
    fresh = rng.normal(SMALL_MEAN, SMALL_STD, size=(4096, 1))
    out = legacy.standardize(fresh)
    assert out.std() < 0.5


# ---------------------------------------------------------------------------
# halflife == 0 is the legacy Welford path, bit-identical
# ---------------------------------------------------------------------------

def test_halflife_zero_is_bitwise_legacy():
    """ema_halflife=0 (and the default) is the legacy all-history Welford path:
    stats and standardize outputs are BIT-identical between a default-constructed
    standardizer and an explicit ema_halflife=0.0 one, and both match the
    offline mean/var exactly."""
    rng = np.random.default_rng(11)
    data = rng.normal(-1.5, 2.5, size=(900, 2))
    default = RunningStandardizer(dim=2)
    explicit = RunningStandardizer(dim=2, ema_halflife=0.0)
    for chunk in np.array_split(data, 9):
        default.update(chunk)
        explicit.update(chunk)

    np.testing.assert_array_equal(default.mean, explicit.mean)
    np.testing.assert_array_equal(default.var, explicit.var)
    assert default.count == explicit.count

    probe = rng.normal(-1.5, 2.5, size=(64, 2)).astype(np.float32)
    np.testing.assert_array_equal(
        default.standardize(probe), explicit.standardize(probe)
    )
    # And the legacy math itself is untouched: exact offline mean/var.
    np.testing.assert_allclose(default.mean, data.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(default.var, data.var(axis=0), rtol=1e-10)


# ---------------------------------------------------------------------------
# Warmup: bias-corrected EMA equals Welford on the first update
# ---------------------------------------------------------------------------

def test_first_update_matches_welford():
    """Bias correction makes the very first update exact: EMA-mode stats after
    one batch equal the Welford stats (the batch mean/var) — no cold-start
    shrinkage toward zero."""
    rng = np.random.default_rng(3)
    batch = rng.normal(1.7, 0.3, size=(128, 1))
    ema = RunningStandardizer(dim=1, ema_halflife=500.0)
    legacy = RunningStandardizer(dim=1)
    ema.update(batch)
    legacy.update(batch)
    np.testing.assert_allclose(ema.mean, legacy.mean, rtol=1e-12)
    np.testing.assert_allclose(ema.var, legacy.var, rtol=1e-12)


def test_bias_correction_exact_on_constant_stream():
    """Feeding the SAME batch repeatedly, the bias-corrected EMA reproduces that
    batch's mean/var exactly at every step (an uncorrected EMA started at zero
    would be scaled by 1 - beta^t ~ 0.004 after 3 updates at halflife 500)."""
    rng = np.random.default_rng(5)
    batch = rng.normal(2.0, 1.5, size=(64, 1))
    ema = RunningStandardizer(dim=1, ema_halflife=500.0)
    for _ in range(3):
        ema.update(batch)
    np.testing.assert_allclose(ema.mean, batch.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(ema.var, batch.var(axis=0), rtol=1e-10)


# ---------------------------------------------------------------------------
# Contract invariants preserved in EMA mode
# ---------------------------------------------------------------------------

def test_ema_identity_before_first_update():
    """count == 0 identity is preserved in EMA mode (same object returned)."""
    ema = RunningStandardizer(dim=1, ema_halflife=100.0)
    x_np = np.array([0.5], dtype=np.float32)
    assert ema.standardize(x_np) is x_np
    x_t = T.tensor([0.5])
    assert ema.standardize(x_t) is x_t


def test_ema_count_tracks_samples():
    """count still counts SAMPLES (the learn-splice contract asserts count grows
    by the batch size per learn step); the half-life clock is UPDATE calls."""
    ema = RunningStandardizer(dim=1, ema_halflife=100.0)
    ema.update(np.zeros((16, 1)))
    ema.update(np.zeros((16, 1)))
    assert ema.count == 32


def test_ema_standardize_is_readonly():
    """standardize never mutates the EMA stats."""
    ema = RunningStandardizer(dim=1, ema_halflife=100.0)
    ema.update(np.array([[1.0], [3.0], [2.0]]))
    m0, v0, c0 = ema.mean.copy(), ema.var.copy(), ema.count
    for _ in range(5):
        ema.standardize(np.array([[10.0], [20.0]]))
        ema.standardize(T.tensor([[10.0], [20.0]]))
    np.testing.assert_array_equal(ema.mean, m0)
    np.testing.assert_array_equal(ema.var, v0)
    assert ema.count == c0


def test_ema_empty_batch_is_noop():
    """An empty batch does not advance the half-life clock or the stats."""
    ema = RunningStandardizer(dim=1, ema_halflife=100.0)
    ema.update(np.array([[1.0], [3.0]]))
    m0, v0, c0 = ema.mean.copy(), ema.var.copy(), ema.count
    ema.update(np.zeros((0, 1)))
    np.testing.assert_array_equal(ema.mean, m0)
    np.testing.assert_array_equal(ema.var, v0)
    assert ema.count == c0


def test_ema_torch_and_numpy_paths_agree():
    """The torch (learn) and numpy (acting) standardize paths agree in EMA mode."""
    ema = RunningStandardizer(dim=2, ema_halflife=100.0)
    ema.update(np.array([[0.1, 5.0], [0.3, 7.0], [0.2, 6.0]]))
    x = np.array([[0.25, 6.5]], dtype=np.float32)
    np_out = ema.standardize(x)
    t_out = ema.standardize(T.tensor(x)).numpy()
    np.testing.assert_allclose(np_out, t_out, rtol=1e-5, atol=1e-6)


def test_negative_halflife_raises():
    with pytest.raises(ValueError):
        RunningStandardizer(dim=1, ema_halflife=-1.0)


# ---------------------------------------------------------------------------
# Actor construction passes the parsed halflife into the standardizer
# ---------------------------------------------------------------------------

_ACTOR_BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.001,
    "LR": 1e-4,
    "EPSILON": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DEC": 1e-4,
    "BATCH_SIZE": 16,
    "MEM_SIZE": 200,
    "REPLACE_TARGET_COUNTER": 100,
    "NOISE": 0.1,
    "UPDATE_ACTOR_ITER": 2,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}


def _make_actor(**overrides):
    from gsp_rl.src.actors.actor import Actor

    config = {**_ACTOR_BASE_CONFIG, **overrides}
    return Actor(
        id=0, config=config, network="DDQN", input_size=31, output_size=9,
        min_max_action=1, meta_param_size=32, gsp=True, gsp_input_size=6,
        gsp_output_size=1,
    )


def test_actor_passes_halflife_into_standardizer():
    actor = _make_actor(
        GSP_E2E_ENABLED=True,
        GSP_E2E_NORMALIZE_FEATURE=True,
        GSP_E2E_NORMALIZE_EMA_HALFLIFE=250,
    )
    assert actor.gsp_e2e_normalize_ema_halflife == 250.0
    assert actor.gsp_feature_stats is not None
    assert actor.gsp_feature_stats.ema_halflife == 250.0


def test_actor_default_halflife_is_legacy_welford():
    actor = _make_actor(GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True)
    assert actor.gsp_e2e_normalize_ema_halflife == 0.0
    assert actor.gsp_feature_stats is not None
    assert actor.gsp_feature_stats.ema_halflife == 0.0
