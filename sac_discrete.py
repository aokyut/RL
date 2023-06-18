import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utills import ConfigParser, huber_error, convert_network_grad_to_false, soft_update, save_model
from tqdm import tqdm
from memory import ReplayMemory
from collections import OrderedDict

SACDiscreteConfig = ConfigParser("SACDiscrete")
SACDiscreteConfig.add_parser("lr_q", 0.0001, float, "learning rate of q network")
SACDiscreteConfig.add_parser("lr_p", 0.0001, float, "learning rate of policy network")
SACDiscreteConfig.add_parser("buffer_size", 10_000, int)
SACDiscreteConfig.add_parser("batch_size", 64, int)
SACDiscreteConfig.add_parser("iter_num", 100_000, int)
SACDiscreteConfig.add_parser("save_n", 10_000, int)
SACDiscreteConfig.add_parser("log_n", 100, int)
SACDiscreteConfig.add_parser("log_name", "sac_test", str)
SACDiscreteConfig.add_parser("tar_update_n", 5, int, "freq of update target network parameters")
SACDiscreteConfig.add_parser("alternating", True, bool)
SACDiscreteConfig.add_parser("gamma", 0.99, float)
SACDiscreteConfig.add_parser("tau", 0.05, float, "soft update parameter")
SACDiscreteConfig.add_parser("tar_ent", 2.0, float)




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
        self.q1_tar = self.q1.clone()
        self.q2_tar = self.q2.clone()
        convert_network_grad_to_false(self.q1_tar)
        convert_network_grad_to_false(self.q2_tar)

        self.memory = ReplayMemory([config["buffer_size"]], config["batch_size"])
        self.config = config
        self.writer = SummaryWriter(log_dir=join("./tensorboard", config["log_name"]))
        self.step = 0
        self.optim_q = optim.Adam(self.q.parameters(), lr=config["lr_q"])
        self.optim_p = optim.Adam(self.p.parameters(), lr=config["lr_p"])

        self.log_n = config["log_n"]
        self.save_n = config["save_n"]
        self.tau = config["tau"]

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=config["lr"])
        self.target_entropy = config["tar_ent"]

    
    def train(self):
        pass

    def optimize(self):
        self.step += 1
        self.p.train()
        self.q.train()
        batch = self.memory.sample()

        state = torch.FloatTensor(batch["state"])
        next_state = torch.FloatTensor(batch["next_state"])
        action = torch.LongTensor(batch["action"]).reshape(-1,1)
        reward = torch.FloatTensor(batch["reward"])
        done = torch.FloatTensor(batch["done"])
        action_mask = torch.FloatTensor(batch["action_mask"])


        alpha = torch.exp(self.log_alpha)

        # q function

        with torch.no_grad():
            next_pi = self.p(next_state, action_mask)
            next_log_pi = torch.log(next_pi + 1e-6)
            next_Q1 = self.q1_tar(next_state)
            next_Q2 = self.q2_tar(next_state)
            next_Q = torch.min(next_Q1, next_Q2)
            if self.config["alternating"]:
                next_Q_tar = reward + (1 - done) * next_pi * (-next_Q - alpha * next_log_pi)
            else:
                next_Q_tar = reward + (1 - done) * next_pi * (next_Q - alpha * next_log_pi)

        td1_error = huber_error(next_Q_tar - Q1)
        td2_error = huber_error(next_Q_tar - Q2)

        td_error = td1_error + td2_error

        q_loss = torch.mean(td_error)

        self.optim_q.zero_grad()
        q_loss.backward()
        self.optim_q.step()

        # policy
        pi = self.p(state)
        log_pi = torch.log(pi + 1e-6)
        Q1 = self.q1(state)
        Q2 = self.q2(state)
        Q = torch.min(Q1, Q2)
        p_loss = (pi * (alpha * log_pi - Q)).mean()

        self.optim_p.zero_grad()
        p_loss.backward()
        self.optim_p.step()

        # alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        


