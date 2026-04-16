"""Tests for ReplayBuffer GSP extensions (co-indexed gsp_obs + gsp_label arrays).

These tests verify the e2e-gsp-training feature where gsp_obs (6 floats) and
gsp_label (1 float, Δθ) are stored alongside each SARSD transition at the same
circular buffer index, enabling joint sampling for end-to-end gradient flow.
"""
import numpy as np
import pytest

from gsp_rl.src.buffers.replay import ReplayBuffer


NUM_OBS = 4
NUM_ACTIONS = 2
GSP_OBS_SIZE = 6
MAX_SIZE = 20
BATCH_SIZE = 8


def make_buffer(gsp_obs_size=0, max_size=MAX_SIZE):
    return ReplayBuffer(
        max_size, NUM_OBS, NUM_ACTIONS, action_type="Continuous",
        gsp_obs_size=gsp_obs_size,
    )


def make_transition(obs_size=NUM_OBS):
    state = np.random.rand(obs_size).astype(np.float32)
    action = np.random.rand(NUM_ACTIONS).astype(np.float32)
    reward = float(np.random.rand())
    state_ = np.random.rand(obs_size).astype(np.float32)
    done = bool(np.random.randint(0, 2))
    return state, action, reward, state_, done


def fill_buffer(buf, n, gsp_obs_size=0):
    for _ in range(n):
        state, action, reward, state_, done = make_transition()
        if gsp_obs_size > 0:
            gsp_obs = np.random.rand(gsp_obs_size).astype(np.float32)
            gsp_label = np.random.rand(1).astype(np.float32)
            buf.store_transition(state, action, reward, state_, done,
                                 gsp_obs=gsp_obs, gsp_label=gsp_label)
        else:
            buf.store_transition(state, action, reward, state_, done)


class TestLegacyPathUnchanged:
    """Buffers constructed without gsp_obs_size must behave identically to the
    original implementation — 5-value return, no new attributes used."""

    def test_legacy_path_unchanged(self):
        buf = make_buffer(gsp_obs_size=0)
        fill_buffer(buf, MAX_SIZE, gsp_obs_size=0)
        result = buf.sample_buffer(BATCH_SIZE)
        assert len(result) == 5, (
            "Legacy buffer must return exactly 5 values: "
            "(states, actions, rewards, next_states, dones)"
        )

    def test_legacy_sample_shapes(self):
        buf = make_buffer(gsp_obs_size=0)
        fill_buffer(buf, MAX_SIZE, gsp_obs_size=0)
        states, actions, rewards, next_states, dones = buf.sample_buffer(BATCH_SIZE)
        assert states.shape == (BATCH_SIZE, NUM_OBS)
        assert actions.shape == (BATCH_SIZE, NUM_ACTIONS)
        assert rewards.shape == (BATCH_SIZE,)
        assert next_states.shape == (BATCH_SIZE, NUM_OBS)
        assert dones.shape == (BATCH_SIZE,)

    def test_legacy_positional_args_not_broken(self):
        """Existing callers pass all 5 args positionally — must still work."""
        buf = make_buffer(gsp_obs_size=0)
        state, action, reward, state_, done = make_transition()
        buf.store_transition(state, action, reward, state_, done)
        assert buf.mem_ctr == 1


class TestE2EStoresAndSamplesGspObsAndLabel:
    """With gsp_obs_size=6, sample_buffer returns 7 values with correct shapes."""

    def test_returns_seven_values(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        fill_buffer(buf, MAX_SIZE, gsp_obs_size=GSP_OBS_SIZE)
        result = buf.sample_buffer(BATCH_SIZE)
        assert len(result) == 7, (
            "GSP buffer must return 7 values: "
            "(states, actions, rewards, next_states, dones, gsp_obs, gsp_labels)"
        )

    def test_gsp_obs_shape(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        fill_buffer(buf, MAX_SIZE, gsp_obs_size=GSP_OBS_SIZE)
        *_, gsp_obs, gsp_labels = buf.sample_buffer(BATCH_SIZE)
        assert gsp_obs.shape == (BATCH_SIZE, GSP_OBS_SIZE)

    def test_gsp_label_shape(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        fill_buffer(buf, MAX_SIZE, gsp_obs_size=GSP_OBS_SIZE)
        *_, gsp_obs, gsp_labels = buf.sample_buffer(BATCH_SIZE)
        assert gsp_labels.shape == (BATCH_SIZE, 1)

    def test_gsp_obs_values_round_trip(self):
        """A specific gsp_obs stored at index 0 must be recoverable."""
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE, max_size=MAX_SIZE)
        known_gsp_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        known_gsp_label = np.array([1.23], dtype=np.float32)
        state, action, reward, state_, done = make_transition()
        buf.store_transition(state, action, reward, state_, done,
                             gsp_obs=known_gsp_obs, gsp_label=known_gsp_label)
        # Verify storage directly (before sampling introduces randomness)
        np.testing.assert_array_almost_equal(buf.gsp_obs_memory[0], known_gsp_obs)
        np.testing.assert_array_almost_equal(buf.gsp_label_memory[0], known_gsp_label)

    def test_gsp_label_values_round_trip(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE, max_size=MAX_SIZE)
        known_gsp_label = np.array([3.14], dtype=np.float32)
        state, action, reward, state_, done = make_transition()
        buf.store_transition(state, action, reward, state_, done,
                             gsp_obs=np.zeros(GSP_OBS_SIZE, dtype=np.float32),
                             gsp_label=known_gsp_label)
        np.testing.assert_array_almost_equal(buf.gsp_label_memory[0], known_gsp_label)


class TestE2EGspObsDefaultsToZeros:
    """When gsp_obs_size > 0 but gsp_obs/gsp_label are not passed, store zeros."""

    def test_gsp_obs_defaults_to_zeros_in_storage(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        state, action, reward, state_, done = make_transition()
        # Call without gsp_obs or gsp_label kwargs
        buf.store_transition(state, action, reward, state_, done)
        np.testing.assert_array_equal(
            buf.gsp_obs_memory[0],
            np.zeros(GSP_OBS_SIZE, dtype=np.float32),
        )

    def test_gsp_label_defaults_to_zeros_in_storage(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        state, action, reward, state_, done = make_transition()
        buf.store_transition(state, action, reward, state_, done)
        np.testing.assert_array_equal(
            buf.gsp_label_memory[0],
            np.zeros(1, dtype=np.float32),
        )

    def test_still_returns_seven_values_even_with_zero_defaults(self):
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        for _ in range(MAX_SIZE):
            state, action, reward, state_, done = make_transition()
            buf.store_transition(state, action, reward, state_, done)
        result = buf.sample_buffer(BATCH_SIZE)
        assert len(result) == 7

    def test_positional_args_not_broken_with_gsp_obs_size_set(self):
        """Existing callers passing 5 positional args must still work when
        gsp_obs_size > 0, even though they don't pass gsp data."""
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE)
        state, action, reward, state_, done = make_transition()
        buf.store_transition(state, action, reward, state_, done)
        assert buf.mem_ctr == 1


class TestCoindexingGspWithMainTransition:
    """gsp_obs[i] in a sampled batch must correspond to state[i] in the same batch."""

    def test_coindexing_gsp_with_main_transition(self):
        """Each stored transition links a unique state value to a unique gsp_obs value.
        After sampling, every (state, gsp_obs) pair in the batch must match the
        originally co-stored pair."""
        buf = make_buffer(gsp_obs_size=GSP_OBS_SIZE, max_size=MAX_SIZE)

        stored_pairs = {}
        for i in range(MAX_SIZE):
            # Use i as a sentinel: state[0] == i, gsp_obs[0] == i + 100
            state = np.full(NUM_OBS, float(i), dtype=np.float32)
            action = np.zeros(NUM_ACTIONS, dtype=np.float32)
            gsp_obs = np.full(GSP_OBS_SIZE, float(i + 100), dtype=np.float32)
            gsp_label = np.array([float(i) * 0.01], dtype=np.float32)
            buf.store_transition(state, action, 0.0, state, False,
                                 gsp_obs=gsp_obs, gsp_label=gsp_label)
            stored_pairs[float(i)] = float(i + 100)

        states, _, _, _, _, gsp_obs_batch, _ = buf.sample_buffer(BATCH_SIZE)

        for row in range(BATCH_SIZE):
            state_sentinel = states[row, 0]
            expected_gsp_sentinel = stored_pairs[state_sentinel]
            actual_gsp_sentinel = gsp_obs_batch[row, 0]
            assert actual_gsp_sentinel == pytest.approx(expected_gsp_sentinel), (
                f"At batch row {row}: state sentinel {state_sentinel} maps to "
                f"gsp_obs sentinel {expected_gsp_sentinel}, got {actual_gsp_sentinel}"
            )
