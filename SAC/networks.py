import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

import numpy as np


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class DualQNetwork(nn.Module):
    def __init__(self,
                 state_shape, action_size,
                 hidden_size, n_blocks):
        super().__init__()
        self.q1 = QNetwork(state_shape, action_size, hidden_size, n_blocks)
        self.q2 = QNetwork(state_shape, action_size, hidden_size, n_blocks)

    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)

        return q1, q2

class QNetwork(nn.Module):
    def __init__(self, 
                 state_shape, action_size: int, 
                 hidden_size: int,
                 n_blocks: int):
        super().__init__()
        self.state_size = 1
        for size in state_shape:
            self.state_size *= size
        self.action_size = action_size
        self.hidden_size = hidden_size
        # self.in_state = nn.Linear(self.state_size, hidden_size)
        # self.in_action = nn.Linear(self.action_size, hidden_size)
        # self.in_block = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.input_layer = nn.Linear(self.state_size + self.action_size, self.hidden_size)
        self.blocks = nn.Sequential()

        for i in range(n_blocks):
            # self.blocks.add_module(
            #     f"LResBlock-{i}", LinearResBlock(hidden_size)
            # )
            self.blocks.add_module(
                f"Linear-{i}", nn.Linear(hidden_size, hidden_size)
            )
            self.blocks.add_module(
                f"rele-{i}", nn.ReLU()
            )
        
        self.out = nn.Linear(self.hidden_size, 1)

    
    def forward(self, state, action):
        """
        Input:
            state: [Batch, ...state_shape]
            action: [Batch, action_size]
        Output:
            q: [Batch, 1]
        """
        state = torch.reshape(state, (-1, self.state_size))
        x = torch.cat([state, action], dim=1)  # [Batch, state_size + action_size]
        x = F.silu(self.input_layer(x))  # [Batch, h_dim]
        x = self.blocks(x)  # [Batch, h_dim]
        x = self.out(x)  # [Batch, 1]

        return x


class LinearResBlock(nn.Module):
    def __init__(self, h_dim:int):
        super().__init__()
        assert h_dim % 4 == 0
        self.layers = nn.Sequential(
            nn.Linear(h_dim, h_dim//4),
            nn.BatchNorm1d(h_dim // 4),
            nn.SiLU(),
            nn.Linear(h_dim//4, h_dim//4),
            nn.SiLU(),
            nn.Linear(h_dim//4, h_dim)
        )
    
    def forward(self, x):
        x = x + self.layers(x)
        return F.silu(x)
    

class GaussianPolicy(nn.Module):
    def __init__(self, state_shape, hidden_size, n_blocks, action_size):
        super().__init__()
        self.state_size = 1
        self.action_bias = 0
        self.action_scale = 2
        for size in state_shape:
            self.state_size *= size
        self.input_layer = nn.Linear(self.state_size, hidden_size)
        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            # self.blocks.add_module(
            #     f"LResBlock-{i}", LinearResBlock(hidden_size)
            # )
            self.blocks.add_module(
                f"Linear-{i}", nn.Linear(hidden_size, hidden_size)
            )
            self.blocks.add_module(
                f"rele-{i}", nn.ReLU()
            )
        self.mu = nn.Linear(hidden_size, action_size)
        self.sig = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """
        Input:
            state: [Batch, ...state_shape]
        Output:
            mu: [Batch, action_size] - [-1 ~ 1]
            sig: [Batch, action_size]
        """
        state = torch.reshape(state, (-1, self.state_size))
        x = F.silu(self.input_layer(state))
        x = self.blocks(x)
        # mu = 2 * torch.tanh(self.mu(x))
        mu = self.mu(x)
        log_std = self.sig(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std

    def sample_action(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mu, std)

        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mu) * self.action_scale + self.action_bias
        # log mu(actions|state)
        # logprobs_gauss = self.__compute_gaussian_logprob(mu, std, actions)
        # actions_squashed = torch.tanh(actions)
        # log pi(actions_squashed|state)
        # logprobs_squashed = logprobs_gauss - torch.sum(torch.log(1 - torch.tanh(actions) ** 2 + 1e-6), dim=1, keepdim=True)

        return action, log_prob
    
    def __compute_gaussian_logprob(self, mean, std, actions):
        """
        Input:
            mean: [Batch, action_size]
            std: [Batch, action_size]
            actions: [Batch, action_size]
        Return:
            logprob: [Batch, 1]
        """
        logprob = - 0.5 * np.log(2 * np.pi)
        logprob += -torch.log(std)
        logprob += - 0.5 * torch.square((actions - mean) / std)
        logprob = torch.sum(logprob, dim=1, keepdim=True)
        return logprob