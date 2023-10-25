import os
import shutil
from os import path
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List
from dataclasses import dataclass
import numpy as np
import torch
import random
import json



@dataclass
class SacConfig:
    state_shape: List[int]
    action_size: int
    buffer_size: int = 2 << 15
    batch_size: int = 64
    log_name: str = "sac"
    log_n: int = 100
    save_n: int = 20000
    iter_n: int = 100000
    n_block: int = 3
    hidden_size: int = 16
    load: bool = False
    save_dir: str = "checkpoint"
    log_dir: str = "tensorboard"
    step: int = 0
    gamma: float = 0.99
    lr_alpha: float = 0.0001
    tar_ent: float = -1
    tar_update_n: int = 8
    tar_update_tau: float = 0.05
    target_hard_update_n: int = 100
    update_per_episode: int = 10
    start_alpha: float = 1.0


def set_summarywriter(config: SacConfig) -> SummaryWriter:
    if config.load:
        # TODO: 
        # load config.step from save dir
        raise Exception()

        # tar_path = path.join(log_dir, log_name)
        # while True:
        #     tar_path = path.join(log_dir, log_name)
    else:
        tar_path = path.join(config.log_dir, config.log_name)
        if path.exists(tar_path):
            print(f"remove {tar_path}")
            shutil.rmtree(tar_path)
        return SummaryWriter(tar_path)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False


def huber_error(td_error)-> torch.Tensor:
    x = torch.abs(td_error)
    return torch.square(x) / (x + 1)


def save_dataclass(tar_path:str, config: SacConfig):
    
    with open(tar_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4, sort_keys=True)

def load_dataclass(tar_path:str, config: SacConfig):
    d = json.loads(tar_path)
    for key, val in d.items():
        config.__dict__[key] = val
    return config

@dataclass
class Transition:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

class ReplayBuffer:
    def __init__(self, config: SacConfig):
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.buffer: List[Transition] = []
        self.idx = 0
        self.is_full = False
        pass

    def push_one_transition(self, exp):
        if self.is_full:
            self.buffer[self.idx] = exp
            self.idx = (self.idx + 1) % self.buffer_size
        else:
            self.buffer.append(exp)
            if len(self.buffer) >= self.buffer_size:
                self.is_full = True

    def push(self, exps: List[Transition]) -> bool:
        """
        Transition: 
            states: np.ndarray(state_shape)
            actions: np.ndarray(action_size)
            next_states: np.ndarray(state_shape)
            rewards: np.ndarray(1)
            dones: np.ndarray(1)
        """
        for exp in exps:
            self.push_one_transition(exp)
        return self.is_full

    def get_batch(self) -> Transition:
        if len(self.buffer) < self.batch_size:
            data_sequence = random.sample(self.buffer, len(self.buffer))
        else:
            data_sequence = random.sample(self.buffer, self.batch_size)
        
        states = np.stack([seq.states for seq in data_sequence], axis=0)
        actions = np.stack([seq.actions for seq in data_sequence])
        next_states = np.stack([seq.next_states for seq in data_sequence])
        rewards = np.stack([seq.rewards for seq in data_sequence])
        dones = np.stack([seq.dones for seq in data_sequence])

        return Transition(
            states, actions, rewards, next_states, dones
        )