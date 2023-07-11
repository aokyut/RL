import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utills import ConfigParser, huber_error, convert_network_grad_to_false, soft_update, save_model
from base.model import Discrete
from tqdm import tqdm
from memory import ReplayMemory
from collections import OrderedDict
from os.path import join
from time import sleep

DqnConfig = ConfigParser("Dqn")

DqnConfig.add_parser("learning_rate", 0.005, float)
DqnConfig.add_parser("buffer_size", 10000, int)
DqnConfig.add_parser("batch_size", 64, int)
DqnConfig.add_parser("iter_num", 100000, int)
DqnConfig.add_parser("save_n", 10000, int)
DqnConfig.add_parser("log_n", 100, int)
DqnConfig.add_parser("log_name", "hogehoge", str)
DqnConfig.add_parser("target_update_n", 1000, int)
DqnConfig.add_parser("soft_update_tau", 0.05, float)
DqnConfig.add_parser("use_sigmoid", True, bool)
DqnConfig.add_parser("alternating", True, bool)
DqnConfig.add_parser("gamma", 0.995, float)

# TODO: explore, trainの実装
# TODO: memoryをDqn内部で実装


class Dqn(nn.Module):
    def __init__(self, network, config, explore_func, eval_func=None):
        super().__init__()
        self.network = network
        self.target_network = network.clone()
        convert_network_grad_to_false(self.target_network)

        self.memory = ReplayMemory(config["buffer_size"], config["batch_size"])
        self.config = config

        self.writer =  SummaryWriter(log_dir=join("./tensorboard", config["log_name"]))
        self.step = 0
        self.optim = optim.Adam(self.network.parameters(), lr=config["learning_rate"])
        self.log_n = config["log_n"]
        self.save_n = config["save_n"]
        self.tau = config["soft_update_tau"]
        self.explore_func = explore_func
        self.eval_func = eval_func
        self.gamma = config["gamma"]

    #train_loop
    def train(self):
        iter_num = self.config["iter_num"]
        bar = tqdm(total=self.config["buffer_size"], desc="[explore]")
        while True:
            data = self.explore_func(self.target_network)
            bar.update(len(data))
            self.memory.push_sequence(data)
            if self.memory.is_full:
                break
        bar = tqdm(total=iter_num, smoothing=0.99)
        while self.step < self.config["iter_num"]:
            # self.target_network.update(self.step)
            data = self.explore_func(self.target_network)
            self.memory.push_sequence(data)
            bar.set_description("[train]")
            loss = self.optimize()
            
            bar.set_postfix(OrderedDict(loss=loss))
            bar.update(1)
            sleep(1)
            
    
    def optimize(self):
        self.step += 1
        self.network.train()
        batch = self.memory.sample()

        state = torch.FloatTensor(batch["state"])
        next_state = torch.FloatTensor(batch["next_state"])
        action = torch.LongTensor(batch["action"]).reshape(-1,1)
        reward = torch.FloatTensor(batch["reward"])
        done = torch.FloatTensor(batch["done"])
        action_mask = torch.FloatTensor(batch["action_mask"])
        if self.config["alternating"]:
            black = torch.FloatTensor(batch["black"]) * -1

        # Q_sa = self.network(state).gather(-1, action.long())
        # with torch.no_grad():
        #     if self.config["alternating"]:
        #         Qtar = -(self.target_network(next_state) - 100 * (1 - action_mask)).max(dim=1).values
        #     else:
        #         Qtar = (self.target_network(next_state) - 100 * (1 - action_mask)).max(dim=1).values

        #     Qtar_sd_ad = reward + self.config["gamma"] * (1 - done) * Qtar
        
        # td_error1 = Qtar_sd_ad - Q_sa
        
        # td_error = huber_error(td_error1)
        # loss = torch.mean(td_error)

        # self.optim.zero_grad()
        # loss.backward()
        # self.optim.step()

        Q = self.network(state).gather(-1, action)
        # Q2 = self.target_network(state).gather(-1, action)

        with torch.no_grad():
            # next_Q1 = self.network(next_state)
            next_Q2 = self.target_network(next_state)
            # next_Q = torch.min(next_Q1, next_Q2)
            if self.config["alternating"]:
                tar_next_Q = black * torch.max(black * next_Q2 - 10000 * (1 - action_mask), dim=-1, keepdim=True).values
            else:
                tar_next_Q = torch.max(next_Q2, dim=-1, keepdim=True).values
            t = reward + (1 - done) * self.gamma * tar_next_Q

        td_error1 = t - Q
        
        td_error = huber_error(td_error1)
        loss = torch.mean(td_error)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.step % self.log_n == 0:
            self.writer.add_scalar("loss/q_loss", loss.item(), self.step)
            if self.eval_func is not None:
                result = self.eval_func(self.target_network)
                for key, val in result.items():
                    self.writer.add_scalar(key, val, self.step)
        
        if self.step % self.save_n == 0:
            save_model(self.step, self.network, self.config["log_name"])
        
        # target_networkの更新
        soft_update(target=self.target_network, source=self.network, tau=self.tau)

        return loss.item()
        



