from base import memory
from collections import namedtuple
import random
import numpy as np

Transition = namedtuple("Transition", ["state", "next_state", "action", "reward", "done", "action_mask", "next_action_mask", "black"])

class ReplayMemory(memory.Memory):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.batch_size = batch_size
        self.flag = False

    @property
    def is_full(self):
        return len(self.buffer) == self.buffer_size
    
    def push_sequence(self, data_seq):
        """
        data: List[
            state: np.array[n]
            action: np.array[n]
            action_mask: np.array[n]
            next_action_mask: np.array[n]
            next_state: np.array[n]
            reward: np.array[1]
            done: np.array[1] , 0 or 1
            ]
        """
        for data in data_seq:
            self.push(data)

    def push(self, data):
        """
        data: 
            state: np.array[n]
            action: np.array[n]
            action_mask: np.array[n]
            next_action_mask: np.array[n]
            next_state: np.array[n]
            reward: np.array[1]
            done: np.array[1] , 0 or 1
            black: np.array[1], 1 or -1
        """
        if "black" not in data:
            data["black"] = np.array([1])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
            self.index += 1
            if self.index >= self.buffer_size:
                self.flag = True
                self.index = 0
    
    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def sample(self):
        data_sequence = random.sample(self.buffer, self.batch_size)

        keys = data_sequence[0].keys()

        batch_dict = {}
        for key in keys:
            batch = []
            for data in data_sequence:
                batch.append(data[key])
            batch_dict[key] = np.stack(batch)

        return batch_dict


class SegmentTree:
    def __init__(self, capacity):
        assert capacity & (capacity - 1) == 0
        self.priorities = [0 for _ in range(2 * capacity)]
        self.capacity = capacity
    
    def __setitem__(self, idx, priority):
        idx = idx + self.capacity
        self.priorities[idx] = priority
        current_idx = idx // 2
        while current_idx >= 1:
            idx_left = current_idx * 2
            idx_right = current_idx * 2 + 1
            self.priorities[current_idx] = self.priorities[idx_left] + self.priorities[idx_right]
            current_idx = current_idx // 2

    def __getitem__(self, idx):
        idx = idx + capacity
        return self.priorities[idx]
    
    def sum(self):
        return self.priorities[1]

    def sample(self):
        z = random.uniform(0, self.sum())
        current_idx = 1
        while current_idx < self.capacity:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            #: 左子ノードよりzが大きい場合は右子ノードへ
            if z > self.priorities[idx_lchild]:
                current_idx = idx_rchild
                z = z -self.priorities[idx_lchild]
            else:
                current_idx = idx_lchild

        priority = self.priorities[current_idx] / self.sum()
        idx = current_idx - self.capacity

        return idx, priority

class PrioritizedReplayMemory:
    def __init__(self, buffer_size, batch_size, alpha):
        self.buffer_size = buffer_size
        self.buffer = []
        self.priorities = SegmentTree(buffer_size)
        self.index = 0
        self.batch_size = batch_size
        self.max_priority = 1
        self.alpha = alpha
    
    @property
    def is_full(self):
        return len(self.buffer)  == self.buffer_size

    def postprocess(self):
        pass

    def push_sequence(self, data_seq):
        """
        data: List[
            state: np.array[n]
            action: np.array[n]
            action_mask: np.array[n]
            next_action_mask: np.array[n]
            next_state: np.array[n]
            reward: np.array[1]
            done: np.array[1] , 0 or 1
            ]
        """
        for data in data_seq:
            self.push(data)
    
    def push(self, data):
        """
        data:
            state: np.array[n]
            action: np.array[n]
            action_mask: np.array[n]
            next_action_mask: np.array[n]
            next_state: np.array[n]
            reward: np.array[1]
            done: np.array[1] , 0 or 1
            black: np.array[1], 1 or-1
        """
        if "black" not in data:
            data["black"] = np.array([1])
        transition = Transition(
            state=data["state"], 
            next_state=data["next_state"], 
            action=data["action"], 
            action_mask=data["action_mask"],
            next_action_mask=data["next_action_mask"],
            reward=data["reward"], 
            done=data["done"],
            black=data["black"]
        )

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
            self.priorities[self.index] = self.max_priority
            self.index = (self.index + 1) %  self.buffer_size
        else:
            self.buffer[self.index] = transition
            self.index = (self.index + 1) %  self.buffer_size
    
    def update_priority(self, td_error):
        new_priorities = (td_error + 1e-6) ** (self.alpha)
        for idx, priority in zip(self.indices, new_priorities):
            self.priorities[idx] = max(priority, 1)
    
    def sample(self):
        indices  = []
        priorities = []
        for i in range(self.batch_size):
            idx, priority = self.priorities.sample()
            priorities.append(priority)
            indices.append(idx)

        priorities = np.array(priorities)

        batch = [self.buffer[idx] for idx in indices]
        batch = [[t.state, t.action, t.reward, t.next_state, t.done, t.action_mask, t.next_action_mask, t.black] for t in batch]
        state, action, reward, next_state, done, action_mask, next_action_mask, black = map(np.stack, zip(*batch))
        self.indices = indices
        return {
            "state": state, 
            "action":action, 
            "reward":reward, 
            "next_state":next_state, 
            "done":done,
            "action_mask": action_mask,
            "next_action_mask": next_action_mask,
            "black": black,
        }