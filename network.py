import torch
import torch.nn as nn
import torch.nn.functional as F
from utills import ConfigParser
from base.model import Network
from torch.distributions import Categorical
from typing import List

import numpy as np
import math


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
            nn.BatchNorm2d(channel // 4),
            nn.SiLU(),
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel // 4),
            nn.SiLU(),
            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
        )
        self.channel = channel
    
    def forward(self, x):
        x = x + self.layers(x)
        return F.silu(x)
    
    def clone(self):
        return BottleNeckBlock(self.channel)


class ResNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int, block_n: int, block_type: type):
        assert issubclass(block_type, Block) and block_type != Block
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.block_n = block_n
        self.block_type = block_type
        self.resblocks = nn.Sequential()
        self.resblocks.add_module(
            "conv2d-input", nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        )
        self.resblocks.add_module(
            "silu-input", nn.SiLU()
        )
        for i in range(1, 1 + block_n):
            self.resblocks.add_module(
                f"resblock-{i}", block_type(hidden_ch)
            )
        self.resblocks.add_module(
            "conv2d-output", nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        )
        self.resblocks.add_module(
            "silu-output", nn.SiLU()
        )

    def forward(self, x):
        return self.resblocks(x)
    
    def clone(self):
        return ResNet(
            self.in_ch, self.out_ch, self.hidden_ch, self.block_n, self.block_type
        )


class PVNet(nn.Module):
    def __init__(self, 
        input_layer: nn.Module,
        policy_layer: nn.Module,
        value_layer: nn.Module, 
        transform = lambda x: x):
        assert issubclass(type(policy_layer), Policy) and type(policy_layer) != Policy
        assert issubclass(type(value_layer), Value) and type(value_layer) != Value
        super().__init__()
        self.input_layer = input_layer
        self.policy = policy_layer
        self.value = value_layer
        self.transform = transform

    def forward(self, state):
        state = self.transform(state)
        x = self.input_layer(state)
        prob = self.policy(x)
        value = self.value(x)
        return prob, value
    
    def clone(self):
        return PVNet(
            self.input_layer.clone(),
            self.policy.clone(),
            self.value.clone(),
            self.transform)


class Policy2d(nn.Module, Policy):
    def __init__(self, in_ch, out_ch, in_fc, out_fc):
        super().__init__()
        self.in_fc = in_fc
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.out_fc = out_fc
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.fc1 = nn.Linear(in_fc, out_fc)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = x.reshape(-1, self.in_fc)
        x = self.fc1(x)
        return F.softmax(x, dim=-1)
    
    def clone(self):
        return Policy2d(self.in_ch, self.out_ch, self.in_fc, self.out_fc)

class Policy1d(nn.Module, Policy):
    def __init__(self, in_feature, out_feature, in_fc, out_fc):
        super().__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.in_fc = in_fc
        self.out_fc = out_fc
        self.linear = nn.Linear(in_feature, out_feature)
        self.out = nn.Linear(in_fc, out_fc)
    def forward(self, x):
        x = F.silu(self.linear(x))
        x = x.reshape(-1, self.in_fc)
        x = self.out(x)
        return F.softmax(x, dim=-1)
    def clone(self):
        return Policy1d(self.in_feature, self.out_feature, self.in_fc, self.out_fc)

class Value2d(nn.Module, Value):
    def __init__(self, in_ch, out_ch, in_fc):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_fc = in_fc
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.fc1 = nn.Linear(in_fc, 1)
    
    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = x.reshape(-1, self.in_fc)
        x = self.fc1(x)
        return x
        # return torch.tanh(x)
    
    def clone(self):
        return Value2d(self.in_ch, self.out_ch, self.in_fc)

class Value1d(nn.Module, Value):
    def __init__(self, in_f, out_f, in_fc):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.in_fc = in_fc
        self.l = nn.Linear(in_f, out_f)
        self.out = nn.Linear(in_fc, 1)

    def forward(self, x):
        x = F.silu(self.l(x))
        x = x.reshape(-1, self.in_fc)
        x = self.out(x)
        return x
    
    def clone(self):
        return Value2d(self.in_f, self.out_f, self.in_fc)

class Attention(nn.Module):
    def __init__(self, depth, q_length, m_length):
        super().__init__()
        self.q_dense = nn.Linear(depth, depth)
        self.k_dense = nn.Linear(depth, depth)
        self.v_dense = nn.Linear(depth, depth)

        self.out_dense = nn.Linear(depth, depth)
        self.scale = 1 / depth ** 0.5
    
    def forward(self, inp, memory):
        """
        query: [b, query_size, depth]
        key: [b, m_size, depth]
        value: [b, m_size, depth]
        """
        query = self.q_dense(inp) / self.scale
        key = self.k_dense(memory)
        value = self.v_dense(memory)

        logit = torch.matmul(query, torch.transpose(key, dim0=-1, dim1=-2))
        attention_weight = torch.softmax(logit, dim=-1)
        print("weight", attention_weight.shape)
        out = self.out_dense(torch.matmul(attention_weight, value))
        return out

class MultiheadSelfAttention(nn.Module, Block):
    def __init__(self, 
        hidden_size:int, 
        head: int, 
        dropout_rate: float,
        length: int
    ):
        super().__init__()
        assert hidden_size % head == 0, ("hidden_size must be divisible by head")

        self.scale = (hidden_size // head) ** -0.5

        self.hidden_size = hidden_size
        self.head = head
        self.length = length
        self.dropout_rate = dropout_rate
        self.q_dense = nn.Linear(hidden_size, hidden_size)
        self.k_dense = nn.Linear(hidden_size, hidden_size)
        self.v_dense = nn.Linear(hidden_size, hidden_size)
        self.out_dense = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(p = dropout_rate)

        self.norm = nn.LayerNorm([length, hidden_size])
    
    def forward(self, x):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)

        q *= self.scale

        logit = torch.matmul(q, torch.transpose(k, dim0=-1, dim1=-2))

        attention_weight = torch.softmax(logit, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)

        attention_out = torch.matmul(attention_weight, v)
        attention_out = self._join(attention_out)
        return self.norm(x + F.silu(self.out_dense(attention_out)))
        # return x + F.silu(self.out_dense(attention_out))

    
    def _split(self, x):
        batch_size, length, hidden_size = x.shape
        x = x.reshape(batch_size, length, self.head, hidden_size // self.head)
        return torch.transpose(x, dim0=2, dim1=1)
    
    def _join(self, x):
        batch_size, head, length, devide_h_size = x.shape
        x = torch.transpose(x, dim0=2, dim1=1)
        return x.reshape(batch_size, length, head * devide_h_size)
    
    def clone(self):
        return MultiheadSelfAttention(self.hidden_size, self.head, self.dropout_rate, self.length)



def get_positional_encoding(max_size: int, d_model: int):
    """
    size: 
        PE(pos, 2i) = sin(pos / 1000 ** (2i / d_model))
        PE(pos, 2i + 1) = cos(pos / 1000 ** (2i / d_model))
    out: 
        (input_size, hidden_size)
    """
    
    mat = np.array(
        [
            [
                (d % 2) * math.sin(pos / 1000 ** (2 * d / d_model))
                + ((d + 1) % 2) * math.cos(pos / 1000 ** (2 * d / d_model))
                for d in range(d_model)
            ]
            for pos in range(max_size)
        ]
    )
    return torch.FloatTensor(mat)

class FeedForwardBlock(nn.Module, Block):
    def __init__(self, h_ch, shape):
        super().__init__()
        self.linear1 = nn.Linear(h_ch, h_ch)
        self.linear2 = nn.Linear(h_ch, h_ch)
        self.norm = nn.LayerNorm(shape)
        self.h_ch = h_ch
        self.shape = shape
    def forward(self, input_x):
        x = self.linear1(input_x)
        x = F.silu(x)
        x = self.linear2(x)
        return self.norm(input_x + x)
        # return input_x + x
    def clone(self):
        return FeedForwardBlock(self.h_ch, self.shape)

class SelfAttentionNet(nn.Module, Block):
    def __init__(self, length: int, hidden_size: int, layers: List[nn.Module], in_ch: int, out_ch: int):
        super().__init__()
        self.positional_encoding = get_positional_encoding(length, hidden_size)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_size = hidden_size
        self.length = length
        self.layers = layers

        self.attention_layers = nn.Sequential()

        self.input_linear = nn.Linear(in_ch, hidden_size)

        for i, layer in enumerate(layers):
            assert issubclass(type(layer), Block) and type(layer) != Block, f"layer: {str(type(layer))}"
            self.attention_layers.add_module(
                f"{(type(layer)).__name__}-{i}", layer.clone()
            )
        
        self.attention_layers.add_module(
            "linear-output", nn.Linear(hidden_size, out_ch)
        )
        self.attention_layers.add_module(
            "silu-output", nn.ReLU()
        )

    def forward(self, x):
        x = F.silu(self.input_linear(x))
        x = x + self.positional_encoding
        return self.attention_layers(x)
    def clone(self):
        return SelfAttentionNet(
            self.length, self.hidden_size, self.layers, self.in_ch, self.out_ch
        )
