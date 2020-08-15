from PPOModel import PPO
from tfConfig import tfConfig

class PPOBrain():
    def __init__(config: Config):
        self.buffer_keys:list= ["state", "action", "action_prob", "reward", "value"]
        
        self.num_agents=num_agents #Number of agents
        self.state_size = config["input_size"]
        self.action_size = config["output_size"]
        self.hidden_units = config["hidden_units"]
        # self.hidden_layer = config["hidden_layer"] # あとで実装現時点では二層
        self.gamma = config["gamma"]
        self.c_vf = config["c_vf"]
        self.beta = config["beta"]
        self.epsilon = config["epsilon"]
        self.lambda_gae = config["lambda"]
        self.time_horizon = config["time_horizon"]
        self.buffer_size = config["buffer_size"]
        self.batch_size = config["batch_size"]
        self.num_epoch = config["num_epoch"]

    def 