import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utills import ConfigParser, huber_error, convert_network_grad_to_false, soft_update, save_model, hard_update
from tqdm import tqdm
from memory import ReplayMemory, PrioritizedReplayMemory
from collections import OrderedDict
from os.path import join
from time import sleep

SACDiscreteConfig = ConfigParser("SACDiscrete")
# SACDiscreteConfig.add_parser("lr_q", 0.005, float, "learning rate of q network")
# SACDiscreteConfig.add_parser("lr_p", 0.005, float, "learning rate of policy network")
SACDiscreteConfig.add_parser("lr_alpha", 0.01, float, "learning rate of policy network")
SACDiscreteConfig.add_parser("buffer_size", 10_000, int)
SACDiscreteConfig.add_parser("batch_size", 32, int)
SACDiscreteConfig.add_parser("iter_num", 100_000, int)
SACDiscreteConfig.add_parser("save_n", 10_000, int)
SACDiscreteConfig.add_parser("log_n", 100, int)
SACDiscreteConfig.add_parser("log_name", "sac_test", str)
SACDiscreteConfig.add_parser("tar_update_n", 5, int, "freq of update target network parameters")
SACDiscreteConfig.add_parser("alternating", True, bool)
SACDiscreteConfig.add_parser("gamma", 0.99, float)
SACDiscreteConfig.add_parser("tau", 0.05, float, "soft update parameter")
SACDiscreteConfig.add_parser("tar_ent", 0.5, float)

SACDiscreteConfig.add_parser("use_per", True, bool, "[prioritized experience replay]: use per memory")
SACDiscreteConfig.add_parser("per_alpha", 0.4, float, "[prioritized experience replay]: weight parameter")





"""
q_network, policy_networkに要求する仕様は最低限で考える。
q_network -> Q(s,a)
policy_network -> π(s, a)
"""

class SACDiscrete(nn.Module):
    def __init__(self, q_network, policy_network, config, explore_func, eval_func=None):
        super().__init__()
        self.q1 = q_network
        self.q2 = q_network.clone()
        self.p = policy_network
        self.p.t = 1
        self.q1_tar = self.q1.clone()
        self.q2_tar = self.q2.clone()
        hard_update(self.q1_tar, self.q1)
        hard_update(self.q2_tar, self.q2)
        convert_network_grad_to_false(self.q1_tar)
        convert_network_grad_to_false(self.q2_tar)

        #優先度付き経験再生
        self.use_per = config["use_per"]
        if self.use_per:
            self.memory = PrioritizedReplayMemory(
                config["buffer_size"],
                config["batch_size"],
                config["per_alpha"])
        else:
            self.memory = ReplayMemory(config["buffer_size"], config["batch_size"])
        self.config = config
        self.writer = SummaryWriter(log_dir=join("./tensorboard", config["log_name"]))
        self.step = 0
        self.optim_q1 = optim.Adam(self.q1.parameters())
        self.optim_q2 = optim.Adam(self.q2.parameters())
        self.optim_p = optim.Adam(self.p.parameters())

        self.log_n = config["log_n"]
        self.save_n = config["save_n"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.optim_alpha = optim.SGD([self.log_alpha], lr=config["lr_alpha"])
        self.target_entropy = config["tar_ent"]

        self.target_update_n = config["tar_update_n"]

        self.explore_func = explore_func
        self.eval_func = eval_func

        self.alternating = config["alternating"]


    
    def train(self):
        iter_num = self.config["iter_num"]
        bar = tqdm(total=self.config["buffer_size"], desc="[explore]")
        while True:
            data = self.explore_func(self.p)
            bar.update(len(data))
            self.memory.push_sequence(data)
            if self.memory.is_full:
                break
        bar = tqdm(total=iter_num, smoothing=0.99)
        while self.step < self.config["iter_num"]:
            # self.target_network.update(self.step)
            data = self.explore_func(self.p)
            self.memory.push_sequence(data)
            bar.set_description("[train]")
            desc = self.optimize()
            
            bar.set_postfix(OrderedDict(desc))
            bar.update(1)
            sleep(0.2)

    def optimize(self):
        self.step += 1
        self.p.train()
        self.q1.train()
        self.q2.train()
        batch = self.memory.sample()

        state = torch.FloatTensor(batch["state"])
        next_state = torch.FloatTensor(batch["next_state"])
        action = torch.LongTensor(batch["action"]).reshape(-1,1)
        reward = torch.FloatTensor(batch["reward"])
        done = torch.FloatTensor(batch["done"])
        action_mask = torch.FloatTensor(batch["action_mask"])
        next_action_mask = torch.FloatTensor(batch["next_action_mask"])
        if self.alternating:
            black = torch.FloatTensor(batch["black"])

        # q function

        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            next_pi = self.p.prob(next_state, next_action_mask)
            next_log_pi = torch.log(next_pi + 1e-6)
            # next_Q1 = self.q1_tar(next_state)
            # next_Q2 = self.q2_tar(next_state)
            next_Q1 = self.q1(next_state)
            next_Q2 = self.q2(next_state)
            next_Q = torch.min(next_Q1, next_Q2)
            
            next_Q_tar = reward + self.gamma * (1 - done) * next_pi * (next_Q - alpha * next_log_pi)
            # print(next_Q_tar.shape)
        Q1 = self.q1(state).gather(-1, action)
        Q2 = self.q2(state).gather(-1, action)

        td1_error = huber_error(next_Q_tar - Q1)
        td2_error = huber_error(next_Q_tar - Q2)

        td_error = td1_error + td2_error

 
        q_loss = torch.mean(td_error)

        self.optim_q1.zero_grad()
        self.optim_q2.zero_grad()
        q_loss.backward()
        self.optim_q1.step()
        self.optim_q2.step()

        # policy
        pi = self.p.prob(state, action_mask)
        log_pi = torch.log(pi + 1e-6)
        with torch.no_grad():
            Q1 = self.q1(state)
            Q2 = self.q2(state)
            Q = torch.min(Q1, Q2)

        if self.alternating:
            p_loss = (pi * (alpha * log_pi - black * Q)).mean()
        else:
            p_loss = (pi * (alpha * log_pi - Q)).mean()

        self.optim_p.zero_grad()
        p_loss.backward()
        self.optim_p.step()

        # alpha
        alpha = torch.exp(self.log_alpha)
        alpha_loss = alpha * (torch.mean(torch.sum(-pi * log_pi, dim=-1)) - self.target_entropy).detach()

        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

        entropy = torch.mean(torch.sum(-pi*log_pi, dim=-1)).item()

        if self.step % self.log_n == 0:
            self.writer.add_scalar("loss/q_loss", q_loss.item(), self.step)
            self.writer.add_scalar("loss/p_loss", p_loss.item(), self.step)
            self.writer.add_scalar("loss/alpha_loss", alpha_loss.item(), self.step)
            self.writer.add_scalar("loss/entropy", entropy, self.step)
            self.writer.add_scalar("loss/alpha", alpha.item(), self.step)

            if self.eval_func is not None:
                result = self.eval_func(self.p)
                for key, val in result.items():
                    self.writer.add_scalar(key, val, self.step)

        if self.step % self.save_n == 0:
            save_model(self.step, self.p, self.config["log_name"])

        if self.step % self.target_update_n == 0:
            hard_update(target=self.q1_tar, source=self.q1)
            hard_update(target=self.q2_tar, source=self.q2)
        # soft_update(target=self.q1_tar, source=self.q1, tau=self.tau)
        # soft_update(target=self.q2_tar, source=self.q2, tau=self.tau)

        
        if self.use_per:
            td_error = td_error.detach().numpy().flatten()
            self.memory.update_priority(td_error)
        

        return {
            "q_loss": q_loss.item(),
            "p_loss": p_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy,
            "alpha_loss": alpha_loss.item()
            }
