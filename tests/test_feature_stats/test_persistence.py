"""Persistence tests for RunningStandardizer save/restore and the
Actor.save_model/load_model wiring.

Motivating incident (2026-07-10): the feature-standardizer stats were never
checkpointed and update() only runs in learn steps, so every fresh-process
eval reconstructed the standardizer cold — standardize() was the identity and
the actor received the raw tiny-scale feature instead of the ~unit-std feature
it was trained on, in every arm of the force-causal-use ablation batch. The
stats are part of the policy; they must round-trip with the checkpoint.

All tests use synthetic numbers only — zero-spend and deterministic.
"""

import numpy as np
import pytest

from gsp_rl.src.actors.feature_stats import RunningStandardizer


def _feed(std: RunningStandardizer, seed: int = 7, updates: int = 40,
          batch: int = 32, mean: float = 0.02, sigma: float = 0.005):
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        std.update(rng.normal(mean, sigma, size=(batch, std.dim)))
    return rng


def _assert_state_equal(a: RunningStandardizer, b: RunningStandardizer):
    assert a.dim == b.dim
    assert a.eps == b.eps
    assert a.count == b.count
    assert a.ema_halflife == b.ema_halflife
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.var, b.var)


def test_welford_round_trip(tmp_path):
    """Welford-mode stats round-trip exactly: the restored instance
    standardizes a probe batch bit-identically to the original."""
    src = RunningStandardizer(dim=3)
    rng = _feed(src)
    p = str(tmp_path / "stats.npz")
    src.save(p)

    dst = RunningStandardizer(dim=3)
    dst.restore(p)
    _assert_state_equal(src, dst)
    probe = rng.normal(0.02, 0.005, size=(64, 3)).astype(np.float32)
    np.testing.assert_array_equal(src.standardize(probe), dst.standardize(probe))


def test_ema_round_trip(tmp_path):
    """EMA-mode stats round-trip exactly, including the halflife/beta and the
    bias-correction update counter (restored stats must keep evolving
    identically if updated further)."""
    src = RunningStandardizer(dim=2, ema_halflife=50.0)
    rng = _feed(src)
    p = str(tmp_path / "stats.npz")
    src.save(p)

    dst = RunningStandardizer(dim=2, ema_halflife=50.0)
    dst.restore(p)
    _assert_state_equal(src, dst)

    # Continue updating both with the same stream: trajectories must not diverge.
    cont = rng.normal(0.02, 0.005, size=(32, 2))
    src.update(cont)
    dst.update(cont)
    np.testing.assert_array_equal(src.mean, dst.mean)
    np.testing.assert_array_equal(src.var, dst.var)


def test_restore_overrides_constructor_numerics(tmp_path):
    """eps and ema_halflife come from the FILE, not the restoring constructor:
    the saved run's numerics define how the policy was trained."""
    src = RunningStandardizer(dim=1, eps=1e-7, ema_halflife=100.0)
    _feed(src)
    p = str(tmp_path / "stats.npz")
    src.save(p)

    dst = RunningStandardizer(dim=1)  # default eps, Welford mode
    dst.restore(p)
    assert dst.eps == 1e-7
    assert dst.ema_halflife == 100.0
    _assert_state_equal(src, dst)


def test_restore_rejects_dim_mismatch(tmp_path):
    """Restoring stats for a different feature width K must raise, not
    silently mis-scale every dimension."""
    src = RunningStandardizer(dim=5)
    _feed(src)
    p = str(tmp_path / "stats.npz")
    src.save(p)

    dst = RunningStandardizer(dim=1)
    with pytest.raises(ValueError, match="dim"):
        dst.restore(p)


def test_restored_cold_stats_stay_identity(tmp_path):
    """Saving a never-updated standardizer and restoring it preserves the
    count==0 identity contract."""
    src = RunningStandardizer(dim=2)
    p = str(tmp_path / "stats.npz")
    src.save(p)
    dst = RunningStandardizer(dim=2)
    dst.restore(p)
    x = np.array([0.5, -0.5], dtype=np.float32)
    assert dst.standardize(x) is x


# ---------------------------------------------------------------------------
# Actor wiring: save_model/load_model round-trips the stats with the networks
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


def test_actor_save_load_round_trips_stats(tmp_path):
    """save_model writes <path>_feature_stats.npz next to the checkpoints and
    a fresh Actor's load_model restores warm stats — the eval-restore fix."""
    actor = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    rng = np.random.default_rng(11)
    for _ in range(20):
        actor.gsp_feature_stats.update(rng.normal(0.02, 0.005, size=(16, 1)))
    assert actor.gsp_feature_stats.count > 0
    ckpt = str(tmp_path / "Episode_10")
    actor.save_model(ckpt)
    assert (tmp_path / "Episode_10_feature_stats.npz").exists()

    fresh = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    assert fresh.gsp_feature_stats.count == 0
    fresh.load_model(ckpt)
    _assert_state_equal(actor.gsp_feature_stats, fresh.gsp_feature_stats)


def test_actor_load_without_stats_file_is_noop(tmp_path):
    """Pre-persistence checkpoints have no stats file: load_model must not
    crash and must leave the stats cold (the RL-CT eval warm-up handles those
    checkpoints)."""
    actor = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    ckpt = str(tmp_path / "Episode_10")
    actor.save_model(ckpt)
    # Simulate a legacy checkpoint: remove the stats file.
    (tmp_path / "Episode_10_feature_stats.npz").unlink()

    fresh = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    fresh.load_model(ckpt)
    assert fresh.gsp_feature_stats.count == 0


def test_actor_load_with_corrupt_stats_file_degrades_to_cold(tmp_path):
    """A truncated/corrupt npz (non-atomic save killed mid-checkpoint) must
    not crash load_model — it degrades to cold stats (the eval warm-up
    fallback), loudly."""
    actor = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    ckpt = str(tmp_path / "Episode_10")
    actor.save_model(ckpt)
    # Truncate the stats file to simulate a mid-write kill.
    stats_file = tmp_path / "Episode_10_feature_stats.npz"
    stats_file.write_bytes(stats_file.read_bytes()[:20])

    fresh = _make_actor(
        GSP_E2E_ENABLED=True, GSP_E2E_NORMALIZE_FEATURE=True,
    )
    fresh.load_model(ckpt)  # must not raise
    assert fresh.gsp_feature_stats.count == 0


def test_actor_without_normalize_flag_saves_no_stats_file(tmp_path):
    """Flag off → gsp_feature_stats is None → save_model writes no stats file
    (byte-identical legacy checkpoint layout)."""
    actor = _make_actor(GSP_E2E_ENABLED=True)
    assert getattr(actor, "gsp_feature_stats", None) is None
    ckpt = str(tmp_path / "Episode_10")
    actor.save_model(ckpt)
    assert not (tmp_path / "Episode_10_feature_stats.npz").exists()
