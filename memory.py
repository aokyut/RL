from base import memory
from collections import namedtuple
import random
import numpy as np

Transition = namedtuple("Transition", ["state", "next_state", "action", "reward", "done", "action_mask"])

class ReplayMemory(memory.Memory):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.batch_size = batch_size
        pass

    @property
    def is_full(self):
        return len(self.buffer) == self.buffer_size
    
    def push_sequence(self, data_seq):
        """
        data: List[
            state: np.array[n]
            action: np.array[n]
            action_mask: np.array[n]
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
            next_state: np.array[n]
            reward: np.array[1]
            done: np.array[1] , 0 or 1
        """
        transition = Transition(state=data["state"], 
            next_state=data["next_state"], 
            action=data["action"], 
            action_mask=data["action_mask"],
            reward=data["reward"], 
            done=data["done"])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
            self.index = (self.index + 1) % self.buffer_size
    
    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        batch = [[t.state, t.action, t.reward, t.next_state, t.done, t.action_mask] for t in batch]
        state, action, reward, next_state, done, action_mask = map(np.stack, zip(*batch))
        return {
            "state": state, 
            "action":action, 
            "reward":reward, 
            "next_state":next_state, 
            "done":done,
            "action_mask": action_mask
        }
