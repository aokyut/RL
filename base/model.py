import torch.nn as nn
import torch
from utils.error import *
import json
from os.path import join
from os import path
import os

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if type(self.config) != dict:
            raise TypeError()
        if "step" not in self.config:
            raise KeyError("step is not in self.config")
        if "save_dir" not in self.config:
            raise KeyError("save_dir is not in self.config")
        if "load_dir" not in self.config:
            self.config["load_dir"] = save_dir
        self.step = 0

    def optimize(self, writer, memory, optimizer):
        raise NotImplementedError()

    @property
    def action_space(self):
        """
        Returns:
        int
        """
        raise NotImplementedError()
    @property
    def observation_space(self):
        """
        Returns:
        tuple()
        """
        raise NotImplementedError()
    def save(self):
        self.config["step"] = self.step
        model_path = join(self.config["save_dir"], self.config["log_name"], "model.pth")
        config_path = join(self.config["save_dir"], self.config["log_name"], "config.json")
        if not path.exists(join(self.config["save_dir"], self.config["log_name"])):
            os.makedirs(join(self.config["save_dir"], self.config["log_name"]))
        torch.save(self.state_dict(), model_path)
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
    def load(self):
        model_path = join(self.config["load_dir"], self.config["log_name"], "model.pth")
        config_path = join(self.config["load_dir"],self.config["log_name"], "config.json")
        if not path.exists(join(self.config["save_dir"], self.config["log_name"])):
            raise ValueError()
        self.load_state_dict(torch.load(model_path))
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.step = config["step"]