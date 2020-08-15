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

        self.local_model = self.brain.get_local_model()
        self.pull_model()

        if train is True:
            self.step_f = self.train_step
        else:
            self.step_f = self.eval_step
    
    def pull_model(self):
        self.local_model.sync(self.brain.model.network)

    def step(self, state, reward, done):
        return self.step_f(state, reward, done)

    def train_step(self, state, reward, done):
        # グローバルモデルからパラメータを取得

        self.buffer_.reward = reward
        
        action = self.local_model.act(state, self.buffer_)

        # final step
        if done is True:
            self.buffer.value = self.buffer_.value
            self.brain.add_memory(self.buffer)
            self.buffer_.value = 0.0
            self.brain.add_memory(self.buffer_)

            # モデルの更新
            self.pull_model()
            return action

        if len(self.buffer) > 0:
            self.buffer.value = self.buffer_.value
            self.brain.add_memory(self.buffer)
        
        self.buffer = self.buffer_
        self.buffer_ = BufferNamespace()
        
        return action

    def eval_step(self, state, reward, done):
        action = self.brain.act(state, self.buffer)

        return action
