import torch.nn as nn
import torch
from utils.error import *
import json
from os.path import join

class BaseModel(nn.Module):
    def __init__(self, config):
        self.config = config
        if type(self.config) != dict:
            raise TypeError()
        if "step" not in self.config:
            raise KeyError("step is not in self.config")
        if "type" not in self.config:
            raise KeyError("type is not in self.config")
        if "save_dir" not in self.config:
            raise KeyError("save_dir is not in self.config")
        if "load_dir" not in self.config:
            self.config["load_dir"] = save_dir

    def loss(self, writer, memory, optimizer):
        raise NotImplementedError()
    @property
    def action_space(self):
        raise NotImplementedError()
    @property
    def observation_space(self):
        raise NotImplementedError()
    def save(self):
        model_path = join(self.config["save_dir"], "model.pth")
        config_path = join(self.config["save_dir"], "config.json")
        torch.save(self.state_dict(), model_path)
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
    def load(self):
        model_path = join(self.config["load_dir"], "model.pth")
        config_path = join(self.config["load_dir"], "config.json")
        self.load_state_dict(torch.load(model_path))
        with open(config_path, "r") as f:
            self.config = json.load(f)