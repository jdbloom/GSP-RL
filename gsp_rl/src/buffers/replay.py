"""Standard SARSD experience replay buffer with circular indexing.

Used by DQN, DDQN, DDPG, TD3, and GSP-DDPG for uniform random experience replay.

See Also: docs/modules/buffers.md
"""
import numpy as np


class ReplayBuffer():
    """Fixed-size circular buffer storing (State, Action, Reward, next_State, Done) tuples.

    Supports both discrete (1D int actions) and continuous (2D float actions)
    action types. Sampling is uniform random without replacement.

    Attributes:
        mem_size: Maximum buffer capacity.
        mem_ctr: Total transitions stored (unbounded, used for circular indexing).
        action_type: 'Discrete' or 'Continuous'.
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
        """
        self.mem_size = max_size
        self.mem_ctr = 0
        self.action_type = action_type
        self.gsp_obs_size = gsp_obs_size
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


    def store_transition(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            state_: np.ndarray,
            done: bool,
            gsp_obs: np.ndarray = None,
            gsp_label: np.ndarray = None,
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
        """
        mem_index = self.mem_ctr % self.mem_size
        self.state_memory[mem_index] = state
        self.action_memory[mem_index] = action
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        if self.gsp_obs_size > 0:
            self.gsp_obs_memory[mem_index] = (
                gsp_obs if gsp_obs is not None
                else np.zeros(self.gsp_obs_size, dtype=np.float32)
            )
            self.gsp_label_memory[mem_index] = (
                gsp_label if gsp_label is not None
                else np.zeros(1, dtype=np.float32)
            )
        self.mem_ctr += 1
        

    def sample_buffer(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a random batch of experiences from the buffer.

        Returns:
            When gsp_obs_size == 0 (legacy path):
                (states, actions, rewards, next_states, dones) — 5 values.
            When gsp_obs_size > 0:
                (states, actions, rewards, next_states, dones, gsp_obs, gsp_labels)
                — 7 values, where gsp_obs and gsp_labels are co-indexed with the
                main transition arrays.
        """
        max_mem = min(self.mem_ctr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)
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