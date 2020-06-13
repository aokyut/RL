import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- PPO discrete model -----
class PPO_discrete_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU()
        )

        self.v_net = nn.Linear(hidden_size, 1)

        self.p_net = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(),
        )
    
    def forward(self, state):
        x = self.pre_net(state)
        v = self.v_net(x)
        p = self.p_net(x)
        return v, p

# ----- PPO continuous model -----
class PPO_continuous_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU()
        )
        self.v_net = nn.Linear(hidden_size, 1)
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        self.sig_net = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
    
    def forward(self, state):
        latent = self.pre_net(state)
        v = self.v_net(latent)
        mu = self.mu_net(latent)
        sig = self.sig_net(latent)
        return v, mu, sig