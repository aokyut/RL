# ----- networks をラッピングして使いやすくする -----
import torch
import numpy as np

from networks import *

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
        
        self.old_network.load_state_dict(self.network.state_dirc())
        self.old_network.to(device)
        self.network.to(device)
        
        # Memory
        self.memory = BufferMemory()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # Loss Function
        self.loss_f = torch.nn.MSELoss()
        
        # Constants
        self.lambda_v = params.lambda_value
        self.lambda_e = params.lambda_entropy
        self.eps = params.epsilon_clip
        self.epoch = params.K_epoch
        self.n_update = params.n_update
        self.device = params.device

        # act
        if params.is_learn is True:
            self.act_f = self.train_act
        else:
            self.act_f = self.eval_act

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
        
        R_t = torch.tensor(R_t).to(device)

        old_states = torch.stack(memory.state).to(self.device).clone()
        old_actions = torch.stack(memory.action).to(self.device).clone()
        old_action_logprobs = torch.stack(memory.action_logprob).to(self.device).clone()

        # optimizing loop
        for i in range(self.epoch):
            # evaluating
            values, policies = self.network(old_states)
            dist = torch.distributions.Categorical(policies)
            action_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            r_theta = torch.exp(logprobs - old_action_logprobs)

            advantage = R_t - values

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages
            loss = - torch.min(surr1, surr2) + self.lambda_v*self.loss_f(state_values, rewards) - self.lambda_e*entropy

            # optimize
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
    
    def add_memory(self, buffer):
        self.memory.action.append(buffer.action)
        self.memory.value_.append(buffer.value)
        self.memory.state.append(buffer.state)
        self.memory.reward.append(buffer.reward)
        self.memory.action_logprob.append(buffer.action_logprob)

        if (len(self.memory) % self.n_update == 0) and (len(self.memory != 0)):
            self.update(self.memory)


