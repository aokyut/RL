import torch
import torch.nn as nn
import torch.nn.functional as F
from utills import ConfigParser
from base.model import Network
from torch.distributions import Categorical

# TODO: ネットワーク構造については別で実装する仕組を考える

DenseConfig = ConfigParser("Dense")
DenseConfig.add_parser("input_size", -1, int)
DenseConfig.add_parser("output_size", -1, int)
DenseConfig.add_parser("hidden_feature_size", 64, int)
DenseConfig.add_parser("hidden_layer_num", 1, int)
DenseConfig.add_parser("use_sigmoid", True, bool)
DenseConfig.add_parser("temperture_init", 0.0, float)
DenseConfig.add_parser("temperture_end", 3.0, float)
DenseConfig.add_parser("temperture_th", 10000, int)


class Dense(nn.Module, Network):
    def __init__(self, config):
        super().__init__()
        self.config = config
        i_size = config["input_size"]
        o_size = config["output_size"]
        h_size = config["hidden_feature_size"]
        h_layers = config["hidden_layer_num"]
        use_sigmoid = config["use_sigmoid"]
        self.input_layer = nn.Linear(in_features=i_size, out_features=h_size)
        self.output_layer = nn.Linear(in_features=h_size, out_features=o_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(h_size, h_size) for i in range(h_layers - 1)
        ])
        self.t =config["temperture_init"]
        self.t_func = lambda step: 3.0 if self.config["temperture_th"] else self.config["temperture_init"] + (self.config["temperture_end"] - self.config["temperture_init"]) * step / self.config["temperture_th"]
    
    def update(self, step):
        self.t = self.t_func(step)
    
    def forward(self, state):
        """
        input: torch.Tensor(batch, input_size)
        output: torch.Tensor(batch, output_size)
        """
        x = self.input_layer(state)
        x = F.leaky_relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.leaky_relu(x)
        return self.output_layer(x)
    
    def get_action(self, state, action_mask, reverse=False):
        x = self.forward(state)
        if reverse:
            probs = (action_mask * F.softmax(-self.t * x, dim=-1))
        else:
            probs = (action_mask * F.softmax(self.t * x, dim=-1))
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    def get_action_eval(self, state, action_mask,reverse=False):
        x = self.forward(state)
        if reverse:
            probs = (action_mask * F.softmax(-x * 10, dim=-1))
        else:
            probs = (action_mask * F.softmax(x * 10, dim=-1))
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    
    def clone(self):
        return Dense(self.config)

DuellingNetworkConfig = ConfigParser("Duelling")
DuellingNetworkConfig.add_parser("input_size", -1, int)
DuellingNetworkConfig.add_parser("output_size", -1, int)
DuellingNetworkConfig.add_parser("hidden_feature_size", 64, int)
DuellingNetworkConfig.add_parser("hidden_layer_num", 1, int)
DuellingNetworkConfig.add_parser("use_sigmoid", True, bool)

class DuellingNetwork:
    def __init__(self, config):
        i_size = config["input_size"]
        o_size = config["output_size"]
        h_size = config["hidden_feature_size"]
        h_layers = config["hidden_layer_num"]
        use_sigmoid = config["use_sigmoid"]
        self.input_layer = nn.Linear(in_features=i_size, out_features=h_size)
        self.output_layer = nn.Linear(in_feature=h_size, out_features=o_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(h_size) for i in range(h_layers - 1)
        ])


CnnConfig = ConfigParser("Cnn")
class Cnn(nn.Module):
    def __init__(self):

        super().__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv2d(4, 6, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(6, 20, kernel_size=3,  stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(20, 100, kernel_size=3, stride=1,padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(100, 226, kernel_size=2, padding=0)
        ])

    def forward(self, state):
        x = state.reshape(-1, 15, 15, 4).transpose(-1, -3)
        for layer in self.cnn_layers:
            x = layer(x)
        return x.reshape(-1, 226)
    
    def get_action(self, state, action_mask, reverse=False):
        x = self.forward(state)
        if reverse:
            probs = (action_mask * F.softmax(-x, dim=-1))
        else:
            probs = (action_mask * F.softmax(x, dim=-1))
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    
    def clone(self):
        return Cnn()