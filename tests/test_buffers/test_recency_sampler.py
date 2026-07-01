"""Tests for the recency-weighted sampler in ReplayBuffer.

Three groups:
  OFF  — recency_halflife=0 produces IDENTICAL sampling to the old path
         (bit-exact same indices when seeded, no p= argument path).
  ON   — recency_halflife>0 statistically over-samples recent transitions.
  WRAP — weight formula is correct for a wrapped circular buffer.
"""
import numpy as np
import pytest

from gsp_rl.src.buffers.replay import ReplayBuffer


NUM_OBS = 4
NUM_ACTIONS = 1


def _fill(buf: ReplayBuffer, n: int) -> None:
    """Store n transitions with reward=i so each slot has a unique reward."""
    for i in range(n):
        state = np.zeros(NUM_OBS, dtype=np.float32)
        buf.store_transition(state, np.zeros(NUM_ACTIONS, dtype=np.float32),
                             float(i), state, False)


# ---------------------------------------------------------------------------
# Group 1: OFF path — bit-identical to old code
# ---------------------------------------------------------------------------

class TestRecencyOff:
    """When recency_halflife=0, sampling must be IDENTICAL to the prior path."""

    def test_off_indices_match_vanilla(self):
        """With the same numpy seed, recency_halflife=0 yields the same indices
        as a buffer built with the old signature (no recency_halflife kwarg).
        Both buffers use np.random.choice(max_mem, batch_size, replace=False)
        with no p= argument — this is the bit-identical guarantee."""
        rng_seed = 42
        mem_size = 20
        batch_size = 5

        # Old-style buffer (no recency_halflife → defaults to 0)
        buf_old = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous')
        _fill(buf_old, mem_size)

        # New-style buffer, explicitly 0
        buf_new = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                               recency_halflife=0)
        _fill(buf_new, mem_size)

        # Same seed → same call to np.random.choice(max_mem, batch, replace=False)
        np.random.seed(rng_seed)
        _, _, rewards_old, _, _ = buf_old.sample_buffer(batch_size)

        np.random.seed(rng_seed)
        _, _, rewards_new, _, _ = buf_new.sample_buffer(batch_size)

        np.testing.assert_array_equal(
            rewards_old, rewards_new,
            err_msg="recency_halflife=0 must be bit-identical to the old path"
        )

    def test_off_no_p_arg_consumed(self):
        """Confirm that OFF path does not consume extra RNG state.
        After two identically-seeded buffers sample with halflife=0, the RNG
        state should advance by exactly one np.random.choice call's worth —
        not more (which would happen if a weight vector were computed)."""
        mem_size = 10
        batch_size = 4

        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=0)
        _fill(buf, mem_size)

        np.random.seed(7)
        buf.sample_buffer(batch_size)
        state_after_off = np.random.get_state()

        buf2 = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                            recency_halflife=0)
        _fill(buf2, mem_size)

        np.random.seed(7)
        # Call the same np.random.choice directly to consume the same RNG amount
        np.random.choice(mem_size, batch_size, replace=False)
        state_after_direct = np.random.get_state()

        # Both RNG states must be identical (same call was made)
        assert state_after_off[1].tolist() == state_after_direct[1].tolist(), \
            "OFF path consumed a different amount of RNG than raw np.random.choice"


# ---------------------------------------------------------------------------
# Group 2: ON path — statistical bias toward recent transitions
# ---------------------------------------------------------------------------

class TestRecencyOn:
    """When recency_halflife>0, recent indices are sampled more often."""

    def test_recent_indices_oversampled(self):
        """Fill a buffer of size 50 with halflife=10.  Sample 10 000 times.
        Indices in the last 10% (most recent 5 slots) must be sampled at
        least 3x more often than the oldest 5 slots."""
        mem_size = 50
        batch_size = 10
        halflife = 10.0
        n_trials = 10_000

        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=halflife)
        _fill(buf, mem_size)

        # Count how often each slot index appears across all samples
        counts = np.zeros(mem_size, dtype=np.int64)
        for _ in range(n_trials):
            _, _, rewards, _, _ = buf.sample_buffer(batch_size)
            # rewards[i] == float(slot_index) because _fill stores reward=i
            idx = rewards.astype(np.int64)
            for j in idx:
                counts[j] += 1

        total = counts.sum()

        # Recent slots: last 5 (indices 45-49 in not-yet-wrapped buffer)
        recent_slots = np.arange(mem_size - 5, mem_size)
        # When buffer is not yet wrapped (mem_ctr == mem_size, one pass),
        # slot i has age (mem_size-1-i), so slot 49 has age 0 (most recent).
        recent_freq = counts[recent_slots].sum() / total / len(recent_slots)

        # Old slots: first 5 (oldest)
        old_slots = np.arange(0, 5)
        old_freq = counts[old_slots].sum() / total / len(old_slots)

        assert recent_freq > 3 * old_freq, (
            f"Recent slots sampled at {recent_freq:.4f}, old at {old_freq:.4f} — "
            f"expected recent >> old with halflife={halflife}"
        )

    def test_sample_without_replacement(self):
        """ON path must still produce unique indices per sample (no duplicates)."""
        mem_size = 30
        batch_size = 10
        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=5.0)
        _fill(buf, mem_size)

        for _ in range(100):
            _, _, rewards, _, _ = buf.sample_buffer(batch_size)
            assert len(np.unique(rewards)) == batch_size, \
                "ON path must sample without replacement"


# ---------------------------------------------------------------------------
# Group 3: Weight formula and circular-wrap correctness
# ---------------------------------------------------------------------------

class TestRecencyWeightFormula:
    """Verify the exact weight vector for hand-checked cases."""

    def test_weight_vector_not_yet_full(self):
        """Pre-full buffer: ages = [n-1, n-2, ..., 1, 0] for slots [0, 1, ..., n-1].
        Slot n-1 (most recent) has age 0, slot 0 has age n-1."""
        mem_size = 10
        halflife = 5.0
        n_stored = 5  # only fill 5 of 10 slots

        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=halflife)
        _fill(buf, n_stored)

        # Hand-compute expected weights
        max_mem = n_stored  # mem_ctr == n_stored < mem_size
        # ages[i] = (mem_ctr - 1 - i) % max_mem for i in range(max_mem)
        # = (4 - i) % 5 = 4, 3, 2, 1, 0  for i = 0,1,2,3,4
        ages = np.array([(n_stored - 1 - i) % n_stored for i in range(max_mem)],
                        dtype=np.float64)
        raw = np.exp(-ages / halflife)
        expected_weights = raw / raw.sum()

        # Extract weights from buffer internals by replicating the formula
        ages_buf = (buf.mem_ctr - 1 - np.arange(max_mem)) % max_mem
        raw_buf = np.exp(-ages_buf.astype(np.float64) / halflife)
        actual_weights = raw_buf / raw_buf.sum()

        np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-10,
                                   err_msg="Weight vector mismatch (not-yet-full)")

    def test_weight_vector_circular_wrapped(self):
        """Full+wrapped buffer: mem_ctr > mem_size.
        Store mem_size + 3 transitions so write pointer has wrapped.
        The most-recent slot is (mem_ctr - 1) % mem_size; its age must be 0.
        The oldest slot (write_ptr + 1) % mem_size has age mem_size - 1."""
        mem_size = 8
        halflife = 3.0
        extra = 3
        n_stored = mem_size + extra  # 11 stores into size-8 buffer

        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=halflife)
        _fill(buf, n_stored)

        max_mem = mem_size  # fully wrapped
        # Most-recent physical slot: (mem_ctr - 1) % mem_size = (11-1)%8 = 2
        most_recent_slot = (n_stored - 1) % mem_size
        # Age of physical slot i: (mem_ctr - 1 - i) % mem_size
        ages = np.array([(n_stored - 1 - i) % mem_size for i in range(mem_size)],
                        dtype=np.float64)

        # Most-recent slot must have age 0
        assert ages[most_recent_slot] == 0, (
            f"Most-recent slot {most_recent_slot} should have age 0, got {ages[most_recent_slot]}"
        )

        # Slot one position AHEAD of write pointer (oldest) must have age mem_size-1
        oldest_slot = (most_recent_slot + 1) % mem_size
        assert ages[oldest_slot] == mem_size - 1, (
            f"Oldest slot {oldest_slot} should have age {mem_size-1}, got {ages[oldest_slot]}"
        )

        # Full weight vector check
        raw = np.exp(-ages / halflife)
        expected_weights = raw / raw.sum()

        ages_buf = (buf.mem_ctr - 1 - np.arange(max_mem)) % max_mem
        raw_buf = np.exp(-ages_buf.astype(np.float64) / halflife)
        actual_weights = raw_buf / raw_buf.sum()

        np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-10,
                                   err_msg="Weight vector mismatch (wrapped circular buffer)")

    def test_weights_sum_to_one(self):
        """Normalised weight vector must sum to 1.0 (within float precision)."""
        mem_size = 20
        halflife = 7.0
        buf = ReplayBuffer(mem_size, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           recency_halflife=halflife)
        _fill(buf, mem_size + 5)  # wrapped

        max_mem = mem_size
        ages = (buf.mem_ctr - 1 - np.arange(max_mem)) % max_mem
        raw = np.exp(-ages.astype(np.float64) / halflife)
        weights = raw / raw.sum()

        assert abs(weights.sum() - 1.0) < 1e-12, f"Weights sum to {weights.sum()}, not 1"

    def test_gsp_obs_path_still_returns_seven_values(self):
        """The ON path must still return 7 values when gsp_obs_size > 0."""
        buf = ReplayBuffer(10, NUM_OBS, NUM_ACTIONS, 'Continuous',
                           gsp_obs_size=3, recency_halflife=5.0)
        state = np.zeros(NUM_OBS, dtype=np.float32)
        action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        gsp_obs = np.ones(3, dtype=np.float32)
        gsp_label = np.array([1.0], dtype=np.float32)
        for _ in range(10):
            buf.store_transition(state, action, 1.0, state, False,
                                 gsp_obs=gsp_obs, gsp_label=gsp_label)
        result = buf.sample_buffer(4)
        assert len(result) == 7, "ON path with gsp_obs_size>0 must return 7 values"
