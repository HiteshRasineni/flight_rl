import numpy as np

class ReplayBuffer:
    """
    Simple ring-buffer replay memory for DQN.
    Stores transitions (s, a, r, s', done).
    """
    def __init__(self, capacity: int, obs_shape):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity,) + self.obs_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        """Return a minibatch (as dict of arrays)."""
        if self.size == 0:
            raise ValueError("Sampling from empty buffer")
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs]
        )

    def __len__(self):
        return self.size
