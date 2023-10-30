from .networks import DualQNetwork, GauusianPolicy
from .utils import *
from typing import List
import math
import torch.optim as optim
from tqdm import tqdm
import sys
from collections import OrderedDict
import os


class GraphSACAgent:
    def __init__(self, config: GraphSacConfig, eval_func, explore_func):
        self.c = config
        self.writer = set_summarywriter(config)
        self.qnet = DualQNetwork(config)
        self.tar_qnet = DualQNetwork(config)
        convert_network_grad_to_false(self.tar_qnet)
        hard_update(self.tar_qnet, self.qnet)

        self.policy = GauusianPolicy(config)

        self.log_alpha = torch.tensor([math.log(config.start_alpha)], requires_grad=True)
        self.optim_alpha = optim.SGD([self.log_alpha], lr=config.lr_alpha)
        self.optim_q = optim.Adam(self.qnet.parameters())
        self.optim_p = optim.Adam(self.policy.parameters())
        self.explore_func = explore_func
        self.eval_func = eval_func

    def train(self):
        bar = tqdm(total=self.c.buffer_size, desc="[warmup]", file=sys.stdout)
        memory = ReplayBuffer(self.c)
        while True:
            data = self.explore_func(self.policy, False)
            bar.update(len(data))
            is_full = memory.push(data)
            if is_full:
                break
        
        bar = tqdm(range(self.c.step // self.c.update_per_episode, self.c.episode_n), smoothing=0.01, desc="[Train]", file=sys.stdout)
        for i in bar:
            self.policy.eval()
            
            data = self.explore_func(self.policy)
            memory.push(data)
            self.policy.train()

            bar2 = tqdm(range(self.c.update_per_episode), smoothing=1, desc="[Optimize]", leave=False, file=sys.stdout)
            

            result = {}
            for j in bar2:
                result = self.optimize(memory.get_batch())
                bar2.set_postfix(result)

            bar.set_postfix(result)
    
    def save(self):
        save_models = {
            "dualq.pth": self.qnet,
            "policy.pth": self.policy,
        }
        save_dir = path.join(self.c.save_dir, self.c.log_name)
        if not path.exists(save_dir):
            os.makedirs(save_dir)
        save_dataclass(path.join(save_dir, "config.json"), self.c)

        alpha_path = path.join(save_dir, "alpha.pt")
        torch.save(self.log_alpha, alpha_path)
        
        for name, model in save_models.items():
            save_path = path.join(save_dir, name)
            torch.save(model.state_dict(), save_path)
    
    def optimize(self, batch: GTransition):
        """
        batch:
            states: [B, N, ...state_size]
            actions: [B, N, action_size]
            rewards: [B, 1, 1]
            next_states: [B, N, ...state_size]
            dones: [B, 1, 1]
            adjs: [B, N, N]
        """
        states = torch.FloatTensor(batch.states)
        actions = torch.FloatTensor(batch.actions)
        rewards = torch.FloatTensor(batch.rewards)
        next_states = torch.FloatTensor(batch.next_states)
        dones = torch.FloatTensor(batch.dones)
        adjs = torch.FloatTensor(batch.adjs)
        edges = torch.FloatTensor(batch.edges)


        with torch.no_grad():
            next_actions, next_logprobs = self.policy.sample_action(adjs, edges, states)
            alpha = torch.exp(self.log_alpha)
            next_q1, next_q2 = self.tar_qnet(adjs, edges, states, actions)
            next_q = torch.min(next_q1, next_q2)
            next_q_tar = rewards + self.c.gamma * (1 - dones) * (next_q - alpha * next_logprobs)
        
        q1, q2 = self.qnet(adjs, edges, states, actions)
        q1_error = huber_error(next_q_tar - q1)
        q2_error = huber_error(next_q_tar - q2)
        # q2_error = F.mse_loss(q2, next_q_tar)
        # q1_error = F.mse_loss(q1, next_q_tar)
        loss_q = torch.mean(0.5 * q1_error + 0.5 * q2_error)

        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # policy
        selected_actions, logprobs = self.policy.sample_action(adjs, edges, states)
        q1, q2 = self.qnet(states, selected_actions)
        q_min = torch.min(q1, q2)

        loss_p = -1 * torch.mean(q_min - alpha * logprobs)

        self.optim_p.zero_grad()
        loss_p.backward()
        self.optim_p.step()

        # alpha
        alpha = torch.exp(self.log_alpha)
        with torch.no_grad():
            selected_actions, logprobs = self.policy.sample_action(adjs, edges, states)
            entropy_diff = -logprobs - self.c.target_entropy
        loss_alpha = torch.mean(alpha * entropy_diff)

        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()

        with torch.no_grad():
            entropy = torch.mean(-logprobs).item()
        
        if self.c.step % self.c.log_n == 0:
            self.writer.add_scalar("loss/loss_q", loss_q.item(), self.c.step)
            self.writer.add_scalar("loss/loss_policy", loss_p.item(), self.c.step)
            self.writer.add_scalar("loss/loss_alpha", loss_alpha.item(), self.c.step)
            self.writer.add_scalar("loss/entropy", entropy, self.c.step)
            self.writer.add_scalar("loss/alpha", alpha.item(), self.c.step)
        
        if self.c.step % self.c.eval_n == 0 and self.eval_func is not None:
            self.policy.eval()
            result = self.eval_func(self.policy)
            for key, val in result.items():
                self.writer.add_scalar(key, val, self.c.step)
            self.policy.train()

        if self.c.step % self.c.save_n == 0:
            self.save()
        
        if self.c.step % self.c.target_update_n == 0:
            soft_update(self.tar_qnet, self.qnet, self.c.target_update_tau)

        self.c.step += 1

        return {
            "q_loss": loss_q.item(),
            "p_loss": loss_p.item(),
            "alpha": alpha.item(),
            "entropy": entropy,
            "alpha_loss": loss_alpha.item()
        }