from models import *

network_type = {
    "ppo_discrete": {
        "global": PPO_discrete,
        "local": Local_ppo_discrete
    }
    "ppo_continuous": {
        "global": PPO_continuous,
        "local": Local_ppo_continuous
    }
}

class Brain:
    def __init__(self, params):
        self.model = network_type[params.model]["global"](params)
        self.params = params
    
    def get_local_model(self):
        return network_type[params.model]["local"](params)
    
    def act(self, state, memory):
        return self.model.act(state, memory)
    
    def add_memory(self, buffer):
        self.model.add_memory(buffer)