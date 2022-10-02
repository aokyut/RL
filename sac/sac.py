import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base.model import BaseModel
from torch.distributions import Normal
import json


class SoftActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_layers, clamp_max, clamp_min):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_size)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)
        ])
        self.mu = nn.Linear(hidden_size, output_dim)
        self.log_sig = nn.Linear(hidden_size, output_dim)
        self.clamp_max = clamp_max
        self.clamp_min = clamp_min
    def forward(self, state):
        x = self.input(state)
        for layer in self.hidden:
            x = layer(x)
        mean = self.mu(x)
        log_std = self.log_sig(x)
        log_std = torch.clamp(log_std, self.clamp_min, self.clamp_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class ClippedCliticNet(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_layers):
        super().__init__()
        self.input1 = nn.Linear(input_dim, hidden_size)
        self.input2 = nn.Linear(input_dim, hidden_size)
        self.hidden1 = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)
        ])
        self.hidden2 = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)
        ])
        self.output1 = nn.Linear(hidden_size, 1)
        self.output2 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.leaky_relu(self.input1(xu))
        for layer in self.hidden1:
            x1 = layer(x1)
            x1 = F.leaky_relu(x1)
        x1 = self.output1(x1)

        x2 = F.leaky_relu(self.input1(xu))
        for layer in self.hidden2:
            x2 = layer(x2)
            x2 = F.leaky_relu(x2)
        x2 = self.output1(x2)
        return x1, x2


class SAC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.target_update_interval = config["target_update_interval"]

        self.actor_net = SoftActorNet(
            input_dim=config["input_dim"], 
            output_dim=config["output_dim"], 
            hidden_size=config['hidden_size'],
            hidden_layers=config["hidden_layers"],
            clamp_max=config["log_std_clamp_max"],
            clamp_min=config["log_std_clamp_min"]
        )
        critic_size = config["input_dim"] + config["output_dim"]
        self.critic_net = ClippedCliticNet(
            input_dim=critic_size,
            hidden_size=config["hidden_size"],
            hidden_layers=config["hidden_layers"]
        )
        self.critic_net_target = ClippedCliticNet(
            input_dim=critic_size,
            hidden_size=config["hidden_size"],
            hidden_layers=config["hidden_layers"]
        )

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=config["lr"])
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=config["lr"])

        self.target_entropy = -torch.prod(torch.Tensor(config["input_dim"])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config["lr"])
        self.save_interval = config["save_interval"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]
        
    @property
    def action_space(self):
        return tuple([self.config["output_dim"]])
    @property
    def observation_space(self):
        return tuple([self.config["input_dim"]])
    
    def get_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if not eval:
            action, _, _ = self.actor_net.sample(state)
        else:
            _, _, action = self.actor_net.sample(state)
        return action.detach().numpy().reshape(-1)

    def optimize(self, writer, memory):
        if len(memory) < self.batch_size:
            return
        self.step += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor_net.sample(next_state_batch)
            next_q1_values_target, next_q2_values_target = self.critic_net_target(next_state_batch, next_action)
            next_q_values_target = torch.min(next_q1_values_target, next_q2_values_target) - self.alpha * next_log_pi
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q1_values, q2_values = self.critic_net(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1_values, next_q_values)
        critic2_loss = F.mse_loss(q2_values, next_q_values)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, log_pi, _ = self.actor_net.sample(state_batch)

        q1_values, q2_values = self.critic_net(state_batch, action)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = ((self.alpha * log_pi) - q_values).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        if self.step % self.config["log_interval"] == 0:
            writer.add_scalar("loss/actor_loss", actor_loss.item(), self.step)
            writer.add_scalar("loss/critic_loss", critic_loss.item(), self.step)
            writer.add_scalar("loss/loss", actor_loss.item() + critic_loss.item(), self.step)
            writer.add_scalar("loss/alpha_loss", alpha_loss.item(), self.step)
            writer.add_scalar("value/q_mean", q_values.mean().item(), self.step)
            writer.add_scalar("value/alpha", self.alpha.item(), self.step)


        if self.step % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)
        
        if self.step % self.save_interval == 0:
            self.save()

        return critic_loss.item(), actor_loss.item()
        


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False
