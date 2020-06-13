# ----- networks をラッピングして使いやすくする -----
import torch
import numpy as np

from networks import *
from utils import debug_text

class BufferMemory:
    def __init__(self):
        self.initialize()
    
    def initialize(self):
        self.state = []
        self.value_ = []
        self.action = []
        self.action_logprob = []
        self.reward = []
    
    def __len__(self):
        return len(self.state)


class PPO_discrete:
    def __init__(self, params):

        # Networks
        self.network = PPO_discrete_network(params.input_size,
                                            params.hidden_size,
                                            params.output_size)
        
        self.old_network = PPO_discrete_network(params.input_size,
                                                params.hidden_size,
                                                params.output_size)
        
        self.old_network.load_state_dict(self.network.state_dict())
        self.old_network.to(params.device)
        self.network.to(params.device)
        
        # Memory
        self.memory = BufferMemory()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=params.lr)

        # Loss Function
        self.loss_f = torch.nn.MSELoss()
        # self.loss_f = torch.nn.SmoothL1Loss()
        
        # Constants
        self.gamma = params.gamma
        self.lambda_v = params.lambda_value
        self.lambda_e = params.lambda_entropy
        self.eps = params.epsilon_clip
        self.epoch = params.K_epoch
        self.n_update = params.n_update
        self.device = params.device

        # act
        if params.is_learn is True:
            self.act_f = self.train_act
            self.network.train()
        else:
            self.act_f = self.eval_act
            self.network.eval()

    def act(self, state, buffer):
        action = self.act_f(state, buffer)
        
        return action

    def eval_act(self, state, buffer):
        v, p = self.network(state)
        dist = torch.distributions.Categorical(p)
        action = dist.sample()
        return action.item()
    
    def train_act(self, state, buffer):
        v, p = self.network(state)
        dist = torch.distributions.Categorical(p)
        action = dist.sample()

        buffer.state = state
        buffer.action = action
        buffer.action_logprob = dist.log_prob(action)
        buffer.value = v

        return action.item()
    
    def update(self, memory):
        # R(t)
        R_t = []
        for reward, value_ in zip(memory.reward, memory.value_):
            R_t.append(reward + self.gamma * value_)
        
        R_t = torch.tensor(R_t).to(self.device).squeeze()

        old_states = torch.stack(memory.state).to(self.device).detach()
        old_actions = torch.stack(memory.action).to(self.device).detach()
        old_action_logprobs = torch.stack(memory.action_logprob).to(self.device).detach()

        # optimizing loop
        for i in range(self.epoch):
            # evaluating
            values, policies = self.network(old_states)
            values = torch.squeeze(values)
            dist = torch.distributions.Categorical(policies)
            action_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            r_theta = torch.exp(action_logprobs - old_action_logprobs.detach())

            advantages = R_t - values.detach()


            surr1 = r_theta * advantages
            surr2 = torch.clamp(r_theta, 1-self.eps, 1+self.eps) * advantages
            L_policy = torch.min(surr1, surr2)
            L_advantage = self.loss_f(values, R_t)
            L_entropy = entropy
            loss = - L_policy + self.lambda_v * L_advantage - self.lambda_e * L_entropy
            loss_mean = loss.mean()
            # optimize
            self.optimizer.zero_grad()
            loss_mean.backward(retain_graph=True)
            self.optimizer.step()
            debug_text(str(loss_mean.item()) + " policy: " +  str(L_policy.mean().item()) + " advantage: " + str(L_advantage.mean().item()) + " entropy: " + str(L_entropy.mean().item()))

        # copy network to old network
        self.old_network.load_state_dict(self.network.state_dict())

    def add_memory(self, buffer):
        self.memory.action.append(buffer.action)
        self.memory.value_.append(buffer.value)
        self.memory.state.append(buffer.state)
        self.memory.reward.append(buffer.reward)
        self.memory.action_logprob.append(buffer.action_logprob)

        if (len(self.memory) % self.n_update == 0) and (len(self.memory) != 0):
            self.update(self.memory)
            self.memory = BufferMemory()

class Local_ppo_discrete:
    def __init__(self, params):
        self.network = PPO_discrete_network(params.input_size,
                                            params.hidden_size,
                                            params.output_size)
        
        self.network.eval()
    
    def act(self, state):
        v, p = self.network(state)
        dist = torch.distributions.Categorical(p)
        action = dist.sample()
        return action.item()
    
    def sync(self, global_network):
        self.network.load_state_dict(global_network.state_dict())


class PPO_continuous:
    def __init__(self, params):
        
        # Netwoks
        self.network = PPO_continuous_network(
            params.input_size,
            params.hidden_size,
            params.output_size
        )

        self.old_network = PPO_continuous_network(
            params.input_size,
            params.hidden_size,
            params.output_size
        )

        # device
        self.device = params.device

        # paste network parameters to old_network
        self.old_network.load_state_dict(self.network.state_dict())
        self.old_network.to(self.device)
        self.network.to(self.device)

        # Memory
        self.memory = BufferMemory()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters)

        # Loss Function
        self.loss_f = torch.nn.MSELoss()

        # Constants
        self.gamma = params.gamma
        self.lambda_v = params.lambda_value
        self.lambda_e = params.lambda_entropy
        self.eps = params.epsilon_clip
        self.epoch = params.K_epoch
        self.n_update = params.n_update
        
        # act setting
        if params.is_learn is True:
            self.act_f = self.train_act
            self.network.train()
        else:
            self.act_f = self.eval_act
            self.network.eval()

    def act(self, state, buffer):
        return self.act_f(state, buffer)

    def eval_act(self, state, buffer):
        v, mu, sig = self.network(state)
        dist = torch.distributions.normal.Normal(mu, sig)
        action = dist.sample()
        return torch.clamp(action.item(), -1.0, 1.0)
    
    def train_act(self, state, buffer):
        v, mu, sig = self.network(state)
        dist = torch.distributions.normal.Normal(mu, sig)
        action = dist.sample()

        buffer.state = state
        buffer.action = action
        buffer.action_logprob = dist.log_prob(action)
        buffer.value = v
        
        return torch.clamp(action.item(), -1.0, 1.0)

    def update(self, memory):
        # R(t)
        R_t = []
        for reward, value_ in zip(memory.reward, memory.value_):
            R_t.append(reward + self.gamma * value_)
        
        R_t = torch.tensor(R_t).to(self.device).squeeze()

        old_states = torch.stack(memory.state).to(self.device).detach()
        old_actions = torch.stack(memory.action).to(self.device).detach()
        old_action_logprobs = torch.stack(memory.action_logprob).to(self.device).detach()

        for i in range(self.epoch):
            values, mus, sigs = self.network(old_states)
            values = torch.squeeze(values)
            dist = torch.distributions.normal.Normal(mus, sigs)
            action_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            r_theta = torch.exp(action_logprobs - old_action_logprobs.detach())

            advantages = R_t - values.detach()

            surr1 = r_theta * advantages
            surr2 = torch.clamp(r_theta, 1-self.eps, 1+self.eps) * advantages
            L_policy = torch.min(surr1, surr2)
            L_advantage = self.loss_f(value, R_t)
            L_entropy = entropy
            loss = - L_policy + self.lambda_v * L_advantage - self.lambda_e * L_entropy
            loss_mean = loss.mean()
            # optimize step
            self.optimizer.zero_grad()
            loss_mean.backward(retain_graph=True)
            self.optimizer.step()
            debug_text(str(loss_mean.item()) + " policy: " + str(L_policy.mean().item()) + " advantage: " + str(L_advantage.mean().item())+ " entropy: " + str(L_entropy.mean().item()))

    def add_memory(self, buffer):
        self.memory.action.append(buffer.action)
        self.memory.value_.append(buffer.value)
        self.memory.state.append(buffer.state)
        self.memory.reward.append(buffer.reward)
        self.memory.action_logprob.append(buffer.action_logprob)

        if (len(self.memory) % self.n_update == 0) and (len(self.memory) != 0):
            self.update(self.memory)
            self.memory = BufferMemory()


class Local_ppo_continuous:
    def __init__(self, params):
        self.network = PPO_continuous_network(
            params.input_size,
            params.hidden_size,
            params.output_size
        )
    
    def act(self, state):
        v, mu, sig = self.network(state)
        dist = torch.distributions.normal.Normal(mu, sig)
        action = dist.sample()
        return torch.clamp(action.item(), -1.0, 1.0)
    
    def sync(self, global_network):
        self.network.load_state_dict(global_network.state_dict())
