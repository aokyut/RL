from dataclasses import dataclass
from typing import List
from os import path
from torch.utils.tensorboard.writer import SummaryWriter
import shutil
import torch
import json 
from argparse import ArgumentParser
import numpy as np
import random

@dataclass
class GraphSacConfig:
    state_shape: List[int]
    edge_shape: List[int]
    action_size: int
    action_bias: float = 0
    action_scale: float = 1
    buffer_size: int = 2 << 15
    batch_size: int = 64
    log_name: str = "gsac"
    log_n: int = 100
    eval_n: int = 100
    save_n: int = 20000
    episode_n: int = 10000
    n_block: int = 3
    v_hidden_size: int = 16
    e_hidden_size: int = 16
    u_hidden_size: int = 16
    load: bool = False
    save_dir: str = "checkpoint"
    log_dir: str = "tensorboard"
    step: int = 0
    gamma: float = 0.99
    lr_alpha: float = 0.0001
    target_entropy: float = -1
    target_update_n: int = 8
    target_update_tau: float = 0.05
    update_per_episode: int = 10
    start_alpha: float = 1.0


def set_summarywriter(config: GraphSacConfig) -> SummaryWriter:
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


def save_dataclass(tar_path:str, config: GraphSacConfig):
    
    with open(tar_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4, sort_keys=True)


def parse_from_dataclass(tar_dclass, default):
    src_dict = tar_dclass.__dataclass_fields__
    parser = ArgumentParser()
    for key, val in src_dict.items():
        default_val = default.__dict__[val.name]
        if val.type != bool:
            parser.add_argument(f"--{val.name}", type=val.type, default=default_val)
        else:
            if val.default:
                parser.add_argument(f"--{val.name}", action="store_false", default=default_val)
            else:
                parser.add_argument(f"--{val.name}", action="store_true", default=default_val)
    tar_dinstance = default
    args = parser.parse_args()
    for key, val in src_dict.items():
        tar_dinstance.__dict__[key] = args.__dict__[key]
    return tar_dinstance


@dataclass
class GTransition:
    adjs: np.ndarray  # [B, N, N]
    edges: np.ndarray  # [B, N, e]
    next_edges: np.ndarray # [B, N, e]
    states: np.ndarray  # [B, N, s]
    actions: np.ndarray  # [B, N, a]
    next_states: np.ndarray  # [B, N, s]
    rewards: np.ndarray  # [B, 1, 1]
    dones: np.ndarray  # [B, 1, 1]

    @classmethod
    def cat(cls, data: List['GTransition']) -> 'GTransition':
        Ns = [t.adjs.shape[2] for t in data]
        N_max = max(Ns)
        adjs = [np.pad(t.adjs, [(0, 0), (0, N_max - N), (0, N_max - N)]) for t, N in zip(data, Ns)]
        edges = [np.pad(t.edges, [(0, 0), (0, N_max - N), (0, 0)]) for t, N in zip(data, Ns)]
        next_edges = [np.pad(t.next_edges, [(0, 0), (0, N_max - N), (0, 0)]) for t, N in zip(data, Ns)]
        states = [np.pad(t.states, [(0, 0), (0, N_max - N), (0, 0)]) for t, N in zip(data, Ns)]
        next_states = [np.pad(t.next_states, [(0, 0), (0, N_max - N), (0, 0)]) for t, N in zip(data, Ns)]
        actions = [np.pad(t.actions, [(0, 0), (0, N_max - N), (0, 0)]) for t, N in zip(data, Ns)]
        dones = [t.dones for t in data]
        rewards = [t.rewards for t in data]
        return cls(
            adjs=np.concatenate(adjs, axis=0),
            edges=np.concatenate(edges, axis=0),
            next_edges=np.concatenate(next_edges, axis=0),
            states=np.concatenate(states, axis=0),
            next_states=np.concatenate(next_states, axis=0),
            actions=np.concatenate(actions, axis=0),
            rewards=np.concatenate(rewards, axis=0),
            dones=np.concatenate(dones, axis=0)
        )

class ReplayBuffer:
    def __init__(self, config:GraphSacConfig):
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.buffer: List[GTransition] = []
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
        pass

    def push(self, exps: List[GTransition]) -> bool:
        """
        Input:
            List[GTransition]:
                adjs: np.ndarray(1, N, N)
                edges: np.ndarray(1, N, e)
                states: np.ndarray(1, N, s)
                actions: np.ndarray(1, N, a)
                next_states: np.ndarray(1, N, s)
                rewards: np.ndarray(1, 1, 1)
                dones: np.ndarray(1, 1, 1)  
        """
        for exp in exps:
            self.push_one_transition(exp)
        return self.is_full
    
    def get_batch(self) -> GTransition:
        if len(self.buffer) < self.batch_size:
            data_sequence = random.sample(self.buffer, len(self.buffer))
        else:
            data_sequence = random.sample(self.buffer, self.batch_size)
        
        return GTransition.cat(data_sequence)
