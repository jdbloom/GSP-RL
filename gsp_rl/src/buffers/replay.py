"""Standard SARSD experience replay buffer with circular indexing.

Used by DQN, DDQN, DDPG, TD3, and GSP-DDPG for uniform random experience replay.
Supports optional recency-weighted sampling (exponential half-life) for gate-training
stabilization — default OFF (recency_halflife=0), bit-identical to the prior implementation.

See Also: docs/modules/buffers.md
"""
import numpy as np


class ReplayBuffer():
    """Fixed-size circular buffer storing (State, Action, Reward, next_State, Done) tuples.

    Supports both discrete (1D int actions) and continuous (2D float actions)
    action types.  Default sampling is uniform random without replacement.

    Optional recency-weighted sampling: when recency_halflife > 0, each valid slot
    is weighted by exp(-age/recency_halflife) where age = (mem_ctr-1-i) % max_mem
    (age 0 = most-recent store).  Handles both the not-yet-full and the wrapped
    circular-buffer cases correctly.  When recency_halflife <= 0 (default), the
    code path is IDENTICAL to the prior implementation — no ``p=`` argument, no
    extra RNG consumption, bit-exact reproducibility for all prior runs.

    Attributes:
        mem_size: Maximum buffer capacity.
        mem_ctr: Total transitions stored (unbounded, used for circular indexing).
        action_type: 'Discrete' or 'Continuous'.
        recency_halflife: Exponential half-life for recency weighting (0 = OFF).
        state_memory: Array of shape (mem_size, num_observations).
        action_memory: Array of shape (mem_size,) for discrete or (mem_size, num_actions) for continuous.
    """
    def __init__(
            self,
            max_size: int,
            num_observations: int,
            num_actions: int,
            action_type: str = None,
            gsp_obs_size: int = 0,
            recency_halflife: float = 0,
            phi_size: int = 0,
    ) -> None:
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
            num_observations: Observation space dimensionality.
            num_actions: Action space dimensionality (1 for discrete).
            action_type: 'Discrete' (int storage) or 'Continuous' (float storage).
            gsp_obs_size: When > 0, allocates parallel gsp_obs_memory of shape
                (max_size, gsp_obs_size) and gsp_label_memory of shape (max_size, 1)
                for co-indexed GSP observation and Δθ label storage.
            recency_halflife: Exponential half-life (in buffer stores) for recency-
                weighted sampling.  0 (default) = OFF: uniform sampling, bit-identical
                to all prior runs.  When > 0, recent transitions are sampled
                exponentially more often — primary stabilizer for target-reset value
                disruption in gate training.
            phi_size: When > 0, allocates a parallel phi_memory of shape
                (max_size, phi_size) for the Successor-Features cumulant phi
                (GSP_SF_ENABLED). 0 (default) = OFF, no allocation, bit-identical
                to all prior runs.
        """
        self.mem_size = max_size
        self.mem_ctr = 0
        self.action_type = action_type
        self.gsp_obs_size = gsp_obs_size
        self.recency_halflife = recency_halflife
        self.phi_size = phi_size
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        if self.action_type == 'Discrete':
            self.action_memory = np.zeros((self.mem_size), dtype = int)
        elif self.action_type == 'Continuous':
            self.action_memory = np.zeros((self.mem_size, num_actions), dtype = np.float32)
        else:
            raise Exception('Unknown Action Type:' + action_type)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool_)
        if self.gsp_obs_size > 0:
            self.gsp_obs_memory = np.zeros((self.mem_size, gsp_obs_size), dtype=np.float32)
            self.gsp_label_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
            # Cache a zero vector to avoid allocating np.zeros on every None-store.
            # Write-protect so any accidental mutation raises immediately rather
            # than silently corrupting future None-stores.
            self._zero_gsp_obs = np.zeros(gsp_obs_size, dtype=np.float32)
            self._zero_gsp_obs.flags.writeable = False
            self._zero_gsp_label = np.zeros(1, dtype=np.float32)
            self._zero_gsp_label.flags.writeable = False
        if self.phi_size > 0:
            self.phi_memory = np.zeros((self.mem_size, phi_size), dtype=np.float32)
            self._zero_phi = np.zeros(phi_size, dtype=np.float32)
            self._zero_phi.flags.writeable = False


    def store_transition(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            state_: np.ndarray,
            done: bool,
            gsp_obs: np.ndarray = None,
            gsp_label: np.ndarray = None,
            phi: np.ndarray = None,
    ) -> None:
        """Store a SARSD experience, optionally with co-indexed GSP data.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Scalar reward received.
            state_: Next observation.
            done: Episode termination flag.
            gsp_obs: GSP-N observation vector (gsp_obs_size floats). When None
                and gsp_obs_size > 0, zeros are stored at this index.
            gsp_label: Δθ label scalar as shape-(1,) array. When None and
                gsp_obs_size > 0, zero is stored at this index.
            phi: Successor-Features cumulant vector (phi_size floats). When None
                and phi_size > 0, zeros are stored at this index.
        """
        mem_index = self.mem_ctr % self.mem_size
        self.state_memory[mem_index] = state
        self.action_memory[mem_index] = action
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        if self.gsp_obs_size > 0:
            self.gsp_obs_memory[mem_index] = (
                gsp_obs if gsp_obs is not None else self._zero_gsp_obs
            )
            self.gsp_label_memory[mem_index] = (
                gsp_label if gsp_label is not None else self._zero_gsp_label
            )
        if self.phi_size > 0:
            self.phi_memory[mem_index] = (
                phi if phi is not None else self._zero_phi
            )
        self.mem_ctr += 1
        

    def sample_buffer(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a batch of experiences from the buffer.

        When recency_halflife <= 0 (OFF): uniform random without replacement —
        IDENTICAL to the prior implementation (no ``p=`` argument, no extra RNG
        consumption).  Existing runs are bit-identical.

        When recency_halflife > 0 (ON): exponential recency weighting.  Age of
        physical slot i is ``(mem_ctr - 1 - i) % max_mem`` so age 0 = most
        recent store.  Weight = exp(-age / recency_halflife), normalised to a
        probability vector over the ``max_mem`` valid indices.

        Returns:
            When gsp_obs_size == 0 (legacy path):
                (states, actions, rewards, next_states, dones) — 5 values.
            When gsp_obs_size > 0:
                (states, actions, rewards, next_states, dones, gsp_obs, gsp_labels)
                — 7 values, where gsp_obs and gsp_labels are co-indexed with the
                main transition arrays.
        """
        batch = self._sample_indices(batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        if self.gsp_obs_size > 0:
            gsp_obs = self.gsp_obs_memory[batch]
            gsp_labels = self.gsp_label_memory[batch]
            return states, actions, rewards, next_states, dones, gsp_obs, gsp_labels

        return states, actions, rewards, next_states, dones

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Draw ``batch_size`` valid slot indices (uniform, or recency-weighted).

        Factored out of :meth:`sample_buffer` so the Successor-Features sampler
        reuses the identical index-selection logic. When recency_halflife <= 0
        this is bit-identical to the prior inline code.
        """
        max_mem = min(self.mem_ctr, self.mem_size)
        if self.recency_halflife <= 0:
            # OFF path: exact prior code — no p= argument, bit-identical.
            return np.random.choice(max_mem, batch_size, replace=False)
        # ON path: exponential recency weighting.
        # ages[i] = number of stores since slot i was written.
        # Unified formula handles both not-yet-full and wrapped cases:
        #   not-yet-full (mem_ctr <= mem_size): ages in [0, mem_ctr-1], no wrap.
        #   full/wrapped (mem_ctr > mem_size): ages mod mem_size, wraps correctly.
        ages = (self.mem_ctr - 1 - np.arange(max_mem)) % max_mem
        raw_weights = np.exp(-ages.astype(np.float64) / self.recency_halflife)
        weights = raw_weights / raw_weights.sum()
        return np.random.choice(max_mem, batch_size, replace=False, p=weights)

    def sample_buffer_sf(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a batch for the Successor-Features learn step (GSP_SF_ENABLED).

        Requires phi_size > 0. Returns the SARSD tuple plus the co-indexed
        cumulant phi:
            (states, actions, rewards, next_states, dones, phi) — 6 values,
            phi of shape (batch_size, phi_size).

        Kept separate from :meth:`sample_buffer` so the legacy len(result)-based
        dispatch in NetworkAids.sample_memory is untouched.
        """
        if self.phi_size <= 0:
            raise ValueError("sample_buffer_sf requires phi_size > 0")
        batch = self._sample_indices(batch_size)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
            self.phi_memory[batch],
        )