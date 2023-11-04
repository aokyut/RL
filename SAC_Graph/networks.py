import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

import numpy as np
from .utils import GraphSacConfig

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

class DualQNetwork(nn.Module):
    def __init__(self, config:GraphSacConfig):
        super().__init__()
        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def forward(self, adj, edge, state, action):
        q1 = self.q1(adj, edge, state, action)
        q2 = self.q2(adj, edge, state, action)
        return q1, q2

class QNetwork(nn.Module):
    def __init__(self, config: GraphSacConfig):
        super().__init__()
        self.state_size = get_size(config.state_shape)
        self.edge_size = get_size(config.edge_shape)
        self.a_size = config.action_size
        self.vh_size = config.v_hidden_size
        self.eh_size = config.e_hidden_size
        self.uh_size = config.u_hidden_size

        self.v_input = nn.Linear(self.state_size + self.a_size, self.vh_size)
        self.e_input = nn.Linear(self.edge_size, self.eh_size)

        self.e_att = nn.Linear(self.eh_size, 1)
        self.v_att = nn.Linear(self.vh_size, 1)
        self.u_input = nn.Linear(self.eh_size + self.vh_size, self.uh_size)

        self.blocks = nn.ModuleList([
            GCNBlock(self.vh_size, self.eh_size, self.uh_size) for i in range(config.n_block)
        ])

        self.out = nn.Linear(self.uh_size, 1)

    def forward(self, adj, edge, state, action):
        """
        Input:
            adj: [Batch, N_max, N_max]
            state: [Batch, N_max, ...state_shape]
            action: [Batch, N_max, action_size]
            edge: [Batch, N_max, ...edge_shape]
        Output:
            q: [Batch, 1, 1]
        """
        B, N, _ = adj.shape
        state = torch.reshape(state, (B, N, self.state_size))
        action = torch.reshape(action, (B, N, self.a_size))
        v = F.relu(self.v_input(torch.cat([state, action], dim=2)))

        edge = torch.reshape(edge, (B, N, self.edge_size))
        e = F.relu(self.e_input(edge))

        e_agg_u = torch.sum(e * F.softmax(self.e_att(e), dim=1), dim=1, keepdim=True)  # [B, 1, eh_size]
        v_agg_u = torch.sum(v * F.softmax(self.v_att(v), dim=1), dim=1, keepdim=True)  # [B, 1, vh_size]
        u = F.relu(self.u_input(torch.cat([e_agg_u, v_agg_u], dim=2)))

        adj_T = F.normalize(torch.transpose(adj, 1, 2), p=1, dim=2)

        for block in self.blocks:
            e, v, u = block(B, N, adj, adj_T, e, v, u)
        
        q = self.out(u)
        return q


class GaussianPolicy(nn.Module):
    def __init__(self, config: GraphSacConfig):
        super().__init__()
        self.state_size = get_size(config.state_shape)
        self.edge_size = get_size(config.edge_shape)
        self.action_bias = config.action_bias
        self.action_scale = config.action_scale
        self.a_size = config.action_size

        self.vh_size = config.v_hidden_size
        self.eh_size = config.e_hidden_size
        self.uh_size = config.u_hidden_size

        self.v_input = nn.Linear(self.state_size, self.vh_size)
        self.e_input = nn.Linear(self.edge_size, self.eh_size)

        self.e_att = nn.Linear(self.eh_size, 1)
        self.v_att = nn.Linear(self.vh_size, 1)
        self.u_input = nn.Linear(self.eh_size + self.vh_size, self.uh_size)
        
        self.blocks = nn.ModuleList([
            GCNBlock(self.vh_size, self.eh_size, self.uh_size) for i in range(config.n_block)
        ])
        self.mu = nn.Linear(self.vh_size, self.a_size)
        self.sig = nn.Linear(self.vh_size, self.a_size)
    
    def forward(self, adj, edge, state):
        """
        Input:
            adj: [Batch, N_max, N_max]
            state: [Batch, N_max, ...state_shape]
            edge: [Batch, N_max, ...edge_shape]
        Output:
            mu: [Batch, N_max * action_size]
            sig: [Batch, N_max * action_size]
        """
        B, N, _ = adj.shape
        state = torch.reshape(state, (B, N, self.state_size))
        v = F.relu(self.v_input(state))

        edge = torch.reshape(edge, (B, N, self.edge_size))
        e = F.relu(self.e_input(edge))

        e_agg_u = torch.sum(e * F.softmax(self.e_att(e), dim=1), dim=1, keepdim=True)  # [B, 1, eh_size]
        v_agg_u = torch.sum(v * F.softmax(self.v_att(v), dim=1), dim=1, keepdim=True)  # [B, 1, vh_size]
        u = F.relu(self.u_input(torch.cat([e_agg_u, v_agg_u], dim=2)))

        adj_T = F.normalize(torch.transpose(adj, 1, 2), p=1, dim=2)

        for block in self.blocks:
            e, v, u = block(B, N, adj, adj_T, e, v, u)
        
        mu = self.mu(v)
        log_std = self.sig(v)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return torch.reshape(mu, (B, -1)), torch.reshape(log_std, (B, -1))
    
    def sample_action(self, adj, edge, state):
        mu, log_std = self.forward(adj, edge, state)
        std = torch.exp(log_std)
        normal = Normal(mu, std)

        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
        # log_prob = log_prob.sum(1, keepdim=True)

        mask = torch.sum(F.normalize(adj, p=1, dim=2), dim=2)
    
        return action * mask, log_prob * mask
        


class GCNBlock(nn.Module):
    def __init__(self, vh_size: int, eh_size: int, uh_size: int):
        super().__init__()
        self.vh_size = vh_size
        self.eh_size = eh_size
        self.uh_size = uh_size
        self.e = nn.Linear(eh_size + 2 * vh_size + uh_size, eh_size)
        self.v = nn.Linear(2 * eh_size + vh_size + uh_size, vh_size)
        self.u = nn.Linear(eh_size + vh_size + uh_size, uh_size)
        self.e_att = nn.Linear(eh_size, 1)
        self.v_att = nn.Linear(vh_size, 1)

    def forward(self, B, N_max, adj, adj_T, e_feature, v_feature, u_feature):
        """
        Input:
            B: int : Batch size
            N_max: int: max node num
            adj: [Batch, N_max, N_max]
            e_feature: [Batch, N_max, eh_size]
            v_feature: [Batch, N_max, vh_size]
            u_feature: [Batch, 1, uh_size]
        """
        v_s = torch.matmul(adj, e_feature)
        u_in = u_feature.expand((B, N_max, self.uh_size))
        e_in = torch.cat([
            e_feature,
            v_feature,
            v_s,
            u_in
        ], dim=2)  # [B, N, eh + 2 * vh + uh]
        e_out = self.e(e_in)  # [B, N, eh]
        e_agg_v = torch.matmul(adj_T, e_out)
        v_in = torch.cat([
            e_agg_v,
            e_out,
            v_feature,
            u_in
        ], dim=2)  # [B, N, 2 *eh + vh + uh]
        v_out = self.v(v_in)  # [B, N, vh]

        e_agg_u = torch.sum(e_out * F.softmax(self.e_att(e_out), dim=1), dim=1, keepdim=True)  # [B, 1, eh_size]
        v_agg_u = torch.sum(v_out * F.softmax(self.v_att(v_out), dim=1), dim=1, keepdim=True)  # [B, 1, vh_size]

        u_in = torch.cat([
            e_agg_u,
            v_agg_u,
            u_feature
        ], dim=2)  # [B, N, eh + vh + uh]
        u_out =  self.u(u_in)
        return F.relu(e_out), F.relu(v_out), F.relu(u_out)



def get_size(state_shape):
    ans_size = 1
    for dim_size in state_shape:
        ans_size *= dim_size
    return ans_size
    