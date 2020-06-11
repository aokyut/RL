from models import *

network_type = {
    "ppo_discrete": PPO_discrete,
}

class Brain:
    def __init__(self, params):
        self.model = network_type[params.model](params)
    
    def act(self, state, memory):
        return self.model.act(state, memory)
    
    def add_memory(self, buffer):
        self.model.add_memory(buffer)