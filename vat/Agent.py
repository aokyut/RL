import torch

class BufferNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __len__(self):
        return len(self.__dict__)


class Agent:
    def __init__(self, brain, train=True):
        self.brain = brain
        self.buffer = BufferNamespace()
        self.buffer_ = BufferNamespace()

        if self.train is True:
            self.step_f = self.train_step
        else:
            self.step_f = self.eval_step

    def step(self, state, reward):
        return self.step_f(state, reward)

    def train_step(self, state, reward):
        self.buffer_.state = state
        self.buffer_.reward = reward
        
        action = self.brain.act(state, self.buffer_)

        if len(self.buffer) > 0:
            self.buffer.value = self.buffer_.value
            self.brain.add_memory(self.buffer)
        
        self.buffer = self.buffer_
        
        return action

    def eval_step(self, state, reward):
        action = self.brain.act(state, self.buffer)

        return action
