"""
Experience Replay Buffer for Policy Gradient Training.

This module provides a replay buffer implementation optimized for the REINFORCE
algorithm used in AzulRL. Unlike DQN-style buffers that store (s, a, r, s', done) tuples,
this buffer stores (state, advantage_weighted_actions) pairs suitable for policy gradient
training with multi-head outputs.

References:
    - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
    - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from collections import deque
import numpy as np
import typing


class ReplayBuffer:
    """
    Experience replay buffer for stabilizing policy gradient training.

    Attributes:
        capacity: Maximum number of experiences to store.
        buffer: Internal deque storing experience tuples.

    The buffer stores experiences as (state, targets) tuples where:
        - state: The observation/state vector
        - targets: List of advantage-weighted action distributions for multi-head models
    """

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store. Older experiences
                     are automatically discarded when capacity is exceeded.
        """
        self.capacity = capacity
        self.buffer: typing.Deque[typing.Tuple[np.ndarray, typing.List[np.ndarray]]] = deque(maxlen=capacity)

    def push(
            self,
            states: np.ndarray,
            targets: typing.List[np.ndarray]
    ) -> None:
        """
        Add experiences to the buffer.

        This method can handle batch insertion of multiple experiences at once.
        Each experience is stored as a separate (state, target) tuple.

        Args:
            states: Array of shape (batch_size, state_dim) containing state observations.
            targets: List of arrays, each of shape (batch_size, action_dim_i),
                    containing advantage-weighted action targets for each model head.
        """
        batch_size = states.shape[0]
        for i in range(batch_size):
            state = states[i]
            target = [t[i] for t in targets]
            self.buffer.append((state, target))

    def sample(
            self,
            batch_size: int
    ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample. If larger than buffer size,
                       samples with replacement.

        Returns:
            Tuple of (states, targets) where:
                - states: Array of shape (batch_size, state_dim)
                - targets: List of arrays for each model head
        """
        replace = batch_size > len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace)

        sampled_states = []
        sampled_targets: typing.List[typing.List[np.ndarray]] = [[] for _ in range(len(self.buffer[0][1]))]

        for idx in indices:
            state, targets = self.buffer[idx]
            sampled_states.append(state)
            for j, target in enumerate(targets):
                sampled_targets[j].append(target)

        states_array = np.array(sampled_states)
        targets_arrays = [np.array(t) for t in sampled_targets]

        return states_array, targets_arrays

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences to start training.

        Args:
            min_size: Minimum number of experiences required.

        Returns:
            True if buffer contains at least min_size experiences.
        """
        return len(self.buffer) >= min_size
