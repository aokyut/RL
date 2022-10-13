import random
import numpy as np


class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0
        self.max_priority = 1.0
        self.indices = []
        self.priorities = []
        self.beta = 0.4
        self.beta_decay = 0.9999
        self.alpha = 0.6
        self.epsilon = 0.01

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.memory_size
    
    def call_beta(self):
        self.beta *= self.beta_decay
        return self.beta

    def sample(self, batch_size):

        probs = np.array(self.priorities) / sum(self.priorities)
        N = len(self.priorities)
        self.indices = np.random.choice(np.arange(N), p=probs, replace=False, size=batch_size)

        beta = self.call_beta()
        weights = (probs[self.indices] * N) ** (-1 * beta)
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)
        
        batch = [self.buffer[idx] for idx in self.indices]


        # batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return weights, state, action, reward, next_state, done
    
    def update_priority(self, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(self.indices, priorities):
            self.priorities[idx] = priority
        
        #:優先度初期値の更新
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return len(self.buffer)