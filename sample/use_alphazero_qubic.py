import torch

import sys
import os

sys.path.append(os.path.abspath(".."))
from envs.qubic import Qubic
from envs.base import get_eval_func, RandomAgent
from network import PVNet, ResNet, BottleNeckBlock, Policy2d, Value2d
from alphazero import PVMCTS


input_net = ResNet(
    in_ch=8, out_ch=64, 
    hidden_ch=128, block_n=3, block_type=BottleNeckBlock,
)

policy_net = Policy2d(
    in_ch=64,
    out_ch=16,
    in_fc=256,
    out_fc=16
)

value_net = Value2d(
    in_ch=64,
    out_ch=16,
    in_fc=256
)


pv = PVNet(input_layer=input_net, 
           policy_layer=policy_net, 
           value_layer=value_net, 
           in_shape=[-1, 8, 4, 4])

env = Qubic()

network = PVMCTS(pv, 0.35, epsilon=0, num_sims=1000, env=env)
state, info = env.reset()
action_mask = info["action_mask"]

while True:
    print(env.render())
    _ = network.analyze_and_action(state, action_mask, False)
    action = int(input())

    next_state, reward, done, _ = env.step(action)
    state = next_state
    
    if done:
        print(env.render())
        break
