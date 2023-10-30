from .networks import DualQNetwork, GaussianPolicy
from .utils import *
from typing import List
from tqdm import tqdm
from collections import OrderedDict
import os
import sys
import math

import torch
import torch.optim as optim
import torch.nn.functional as F

# from torch.utils.tensorboard.writer import SummaryWriter



class SACAgent:
    def __init__(self, config: SacConfig, eval_func, explore_func):
        self.eval_func = eval_func
        self.explore_func = explore_func
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.episode_n = config.episode_n
        self.save_n = config.save_n
        self.log_n = config.log_n
        self.eval_n = config.eval_n
        self.writer = set_summarywriter(config)
        self.step = config.step
        self.log_name = config.log_name
        self.save_dir = config.save_dir
        self.target_entropy = config.tar_ent
        self.tar_update_n = config.tar_update_n
        self.tar_hard_update_n = config.target_hard_update_n
        self.update_per_episode = config.update_per_episode
        self.optimize_n = config.update_per_episode
        self.tau = config.tar_update_tau
        self.qnet = DualQNetwork(
            state_shape=config.state_shape,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            n_blocks=config.n_block
        )
        self.tar_qnet = DualQNetwork(
            state_shape=config.state_shape,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            n_blocks=config.n_block
        )
        convert_network_grad_to_false(self.tar_qnet)
        hard_update(self.tar_qnet, self.qnet)

        self.policy = GaussianPolicy(
            state_shape=config.state_shape,
            hidden_size=config.hidden_size,
            n_blocks=config.n_block,
            action_size=config.action_size,
            action_scale=config.action_scale,
            action_bias=config.action_bias
        )
        self.gamma = config.gamma
        # self.log_alpha = torch.zeros(1, requires_grad=True)
        self.log_alpha = torch.tensor([math.log(config.start_alpha)], requires_grad=True)
        self.optim_alpha = optim.SGD([self.log_alpha], lr=config.lr_alpha)
        self.optim_q = optim.Adam(self.qnet.parameters())
        self.optim_p = optim.Adam(self.policy.parameters())
        self.config = config
    
    def train(self):
        bar = tqdm(total=self.config.buffer_size, desc="[warm up]", file=sys.stdout)
        memory = ReplayBuffer(self.config)
        while True:
            data = self.explore_func(self.policy, False)
            bar.update(len(data))
            is_full = memory.push(data)
            if is_full:
                break
        
        bar = tqdm(range(self.step // self.update_per_episode, self.episode_n), smoothing=0.01, desc="[Train]", file=sys.stdout)
        for i in bar:
            self.policy.eval()
            
            data = self.explore_func(self.policy)
            memory.push(data)
            self.policy.train()

            bar2 = tqdm(range(self.optimize_n), smoothing=1, desc="[Optimize]", leave=False, file=sys.stdout)

            for j in bar2:
                result = self.optimize(memory.get_batch())
                bar2.set_postfix(OrderedDict(result))

            bar.set_postfix(OrderedDict(result))
    
    def save(self):
        save_models = {
            "dualq.pth": self.qnet,
            "policy.pth": self.policy,
        }
        save_dir = path.join(self.save_dir, self.log_name)
        if not path.exists(save_dir):
            os.makedirs(save_dir)
        self.config.step = self.step
        save_dataclass(path.join(save_dir, "config.json"), self.config)

        alpha_path = path.join(save_dir, "alpha.pt")
        torch.save(self.log_alpha, alpha_path)
        
        for name, model in save_models.items():
            save_path = path.join(save_dir, name)
            torch.save(model.state_dict(), save_path)
    
    def load(self):
        # TODO
        pass

    def optimize(self, batch: Transition):
        """
        batch: 
            states: [B, ...state_shape]
            actions: [B, action_size]
            rewards: [B, 1]
            next_states: [B, ...state_shape]
            dones: [B, 1]
        """
        states = torch.FloatTensor(batch.states)
        actions = torch.FloatTensor(batch.actions)
        rewards = torch.FloatTensor(batch.rewards)
        next_states = torch.FloatTensor(batch.next_states)
        dones = torch.FloatTensor(batch.dones)


        with torch.no_grad():
            next_actions, next_logprobs = self.policy.sample_action(next_states)
            alpha = torch.exp(self.log_alpha)
            next_q1, next_q2 = self.tar_qnet(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            next_q_tar = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_logprobs)
        
        q1, q2 = self.qnet(states, actions)
        q1_error = huber_error(next_q_tar - q1)
        q2_error = huber_error(next_q_tar - q2)
        # q2_error = F.mse_loss(q2, next_q_tar)
        # q1_error = F.mse_loss(q1, next_q_tar)
        loss_q = torch.mean(0.5 * q1_error + 0.5 * q2_error)

        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # policy
        selected_actions, logprobs = self.policy.sample_action(states)
        q1, q2 = self.qnet(states, selected_actions)
        q_min = torch.min(q1, q2)

        loss_p = -1 * torch.mean(q_min - alpha * logprobs)

        self.optim_p.zero_grad()
        loss_p.backward()
        self.optim_p.step()

        # alpha
        alpha = torch.exp(self.log_alpha)
        with torch.no_grad():
            selected_actions, logprobs = self.policy.sample_action(states)
            entropy_diff = -logprobs - self.target_entropy
        loss_alpha = torch.mean(alpha * entropy_diff)

        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()

        with torch.no_grad():
            entropy = torch.mean(-logprobs).item()
        
        if self.step % self.log_n == 0:
            self.writer.add_scalar("loss/loss_q", loss_q.item(), self.step)
            self.writer.add_scalar("loss/loss_policy", loss_p.item(), self.step)
            self.writer.add_scalar("loss/loss_alpha", loss_alpha.item(), self.step)
            self.writer.add_scalar("loss/entropy", entropy, self.step)
            self.writer.add_scalar("loss/alpha", alpha.item(), self.step)
        
        if self.step % self.eval_n == 0 and self.eval_func is not None:
            self.policy.eval()
            result = self.eval_func(self.policy)
            for key, val in result.items():
                self.writer.add_scalar(key, val, self.step)
            self.policy.train()

        if self.step % self.save_n == 0:
            self.save()
        
        if self.step % self.tar_update_n == 0:
            soft_update(self.tar_qnet, self.qnet, self.tau)
        
        if self.step % self.tar_hard_update_n == 0:
            hard_update(self.tar_qnet, self.qnet)
        
        self.step += 1

        return {
            "q_loss": loss_q.item(),
            "p_loss": loss_p.item(),
            "alpha": alpha.item(),
            "entropy": entropy,
            "alpha_loss": loss_alpha.item()
        }