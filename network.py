import torch
import torch.nn as nn
import torch.nn.functional as F
from utills import ConfigParser
from base.model import Network
from torch.distributions import Categorical
from typing import List


class Block:
    pass

class Policy:
    pass

class Value:
    pass

class Q:
    pass



DenseConfig = ConfigParser("Dense")
DenseConfig.add_parser("input_size", -1, int)
DenseConfig.add_parser("output_size", -1, int)
DenseConfig.add_parser("hidden_feature_size", 64, int)
DenseConfig.add_parser("hidden_layer_num", 1, int)
DenseConfig.add_parser("use_sigmoid", True, bool)
DenseConfig.add_parser("temperture_init", 0.0, float)
DenseConfig.add_parser("temperture_end", 3.0, float)
DenseConfig.add_parser("temperture_th", 10000, int)


# TODO: sacとdqnで値の扱い方が変わるのでネットワークの扱いを変える事を考える。

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
    
    def prob(self, state, action_mask):
        x = self.forward(state)
        # probs = (action_mask * F.softmax(self.t * x, dim=-1))
        probs = (action_mask * (F.softmax(x, dim=-1) + 1e-6))
        return probs

    def get_action(self, state, action_mask, reverse=False):
        reverse = False
        x = self.forward(state)
        if reverse:
            probs = (action_mask * (F.softmax(-x, dim=-1) + 1e-6))
        else:
            probs = (action_mask * (F.softmax(x, dim=-1) + 1e-6))
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def get_action_eval(self, state, action_mask,reverse=False):
        reverse = False
        x = self.forward(state)
        if reverse:
            return torch.max(action_mask * (F.softmax(-x, dim=-1) + 1e-6), dim=-1)[1].item()
        return torch.max(action_mask * (F.softmax(x, dim=-1) + 1e-6), dim=-1)[1].item()
    
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
            nn.Conv2d(2, 6, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(6, 20, kernel_size=3,  stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(20, 100, kernel_size=3, stride=1,padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(100, 451, kernel_size=2, padding=0)
        ])

    def forward(self, state):
        x = state.reshape(-1, 15, 15, 2).transpose(-1, -3)
        for layer in self.cnn_layers:
            x = layer(x)
        return x.reshape(-1, 451)
    
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

# BottleNeck
class BottleNeckBlock(nn.Module, Block):
    def __init__(self, channel:int):
        assert channel % 4 == 0
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            # nn.BatchNorm2d(channel // 4),
            nn.ReLU(),
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(channel // 4),
            nn.ReLU(),
            nn.Conv2d(channel // 4, channel, kernel_size=1)
        )
    
    def forward(self, x):
        x = x + self.layers(x)
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int, block_n: int, block_type: type):
        assert issubclass(block_type, Block) and block_type != Block
        super().__init__()
        self.resblocks = nn.Sequential()
        self.resblocks.add_module(
            "conv2d-input", nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        )
        self.resblocks.add_module(
            "relu-input", nn.ReLU()
        )
        for i in range(1, 1 + block_n):
            self.resblocks.add_module(
                f"resblock-{i}", block_type(hidden_ch)
            )
        self.resblocks.add_module(
            "conv2d-output", nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        )
        self.resblocks.add_module(
            "relu-output", nn.ReLU()
        )

    def forward(self, x):
        return self.resblocks(x)


class PVNet(nn.Module):
    def __init__(self, 
        input_layer: nn.Module,
        policy_layer: nn.Module,
        value_layer: nn.Module, 
        in_shape: List[int]):
        assert issubclass(type(policy_layer), Policy) and type(policy_layer) != Policy
        assert issubclass(type(value_layer), Value) and type(value_layer) != Value
        super().__init__()
        self.input_layer = input_layer
        self.policy = policy_layer
        self.value = value_layer
        self.in_shape = in_shape

    def forward(self, state):
        state = state.reshape(self.in_shape)
        x = self.input_layer(state)
        prob = self.policy(x)
        value = self.value(x)
        return prob, value


class Policy2d(nn.Module, Policy):
    def __init__(self, in_ch, out_ch, in_fc, out_fc):
        super().__init__()
        self.in_fc = in_fc
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.fc1 = nn.Linear(in_fc, out_fc)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(-1, self.in_fc)
        x = self.fc1(x)
        return F.softmax(x, dim=-1)

class Value2d(nn.Module, Value):
    def __init__(self, in_ch, out_ch, in_fc):
        super().__init__()
        self.in_fc = in_fc
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.fc1 = nn.Linear(in_fc, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(-1, self.in_fc)
        x = self.fc1(x)
        return torch.tanh(x)
