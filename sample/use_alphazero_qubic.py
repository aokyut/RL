import torch

import sys
import os

sys.path.append(os.path.abspath(".."))
from envs.qubic import Qubic
from envs.base import get_eval_func, RandomAgent, MiniMaxAgent, compare_agents, ModelAgent
from network import PVNet, ResNet, BottleNeckBlock, Policy2d, Value2d
from alphazero import PVMCTS

input_net = ResNet(
    in_ch=8, out_ch=16, 
    hidden_ch=16, block_n=4, block_type=BottleNeckBlock,
)

policy_net = Policy2d(
    in_ch=16,
    out_ch=8,
    in_fc=128,
    out_fc=16
)

value_net = Value2d(
    in_ch=16,
    out_ch=8,
    in_fc=128
)


pv = PVNet(input_layer=input_net, 
           policy_layer=policy_net, 
           value_layer=value_net, 
           in_shape=[-1, 8, 4, 4])

pv.load_state_dict(torch.load("../checkpoint/alphazero_qubic/latest.pth"))

from main import eval_func

network = PVMCTS(pv, 0.35, epsilon=0, num_sims=50, env=env)
print(eval_func(network))

exit(0)

env = Qubic()

state, info = env.reset()
action_mask = info["action_mask"]
agents = [
    RandomAgent(),
    MiniMaxAgent(1),
    MiniMaxAgent(0),
    # ModelAgent(network)
]
compare_agents(env = env, agents=agents, eval_n=5)

while True:
    print(env.render())
    _ = network.analyze_and_action(state, action_mask, False)
    action = int(input())

    next_state, reward, done, _ = env.step(action)
    state = next_state
    
    if done:
        print(env.render())
        break
