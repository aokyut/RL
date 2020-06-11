import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- PPO discrete model -----
class PPO_discrete_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super.__init__()
        self.dense_in = nn.Linear(opts.input_size, opts.hidden_size)
        self.dropout = nn.Dropout

        self.pre_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.v_net = nn.Linear(hidden_size, 1)

        self.p_net = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(),
        )
    
    def forward(self, state):
        x = self.pre_net(state)
        return self.v_net(x), self.p_net(x)

# ----- PPO continuous model -----
class PPO_continuous_network(nn.Module):
    def __init__(self):
        pass