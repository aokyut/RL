import torch

import sys
import os

sys.path.append(os.path.abspath(".."))
from envs.ox_alpha import OXEnv
from envs.base import get_eval_func, RandomAgent, MiniMaxAgent, compare_agents
from network import PVNet, ResNet, BottleNeckBlock, Policy2d, Value2d
from alphazero import PVMCTS


input_net = ResNet(
    in_ch=2, out_ch=4, 
    hidden_ch=8, block_n=2, block_type=BottleNeckBlock,
)

policy_net = Policy2d(
    in_ch=4,
    out_ch=2,
    in_fc=18,
    out_fc=9
)

value_net = Value2d(
    in_ch=4,
    out_ch=2,
    in_fc=18
)

pv = PVNet(
    input_layer=input_net,
    policy_layer=policy_net,
    value_layer=value_net,
    in_shape=[-1, 2, 3, 3])

# pv.load_state_dict(torch.load("../checkpoint/alphazero_ox_debug5/latest.pth"))

env = OXEnv()

state, info = env.reset()
action_mask = info["action_mask"]

network = PVMCTS(pv, 0.35, epsilon=0, num_sims=10, env=env)

class NewAgent:
    def __init__(self, pv, env):
        self.network = pv
        self.env = env
    def get_action_eval(self, state, action_mask, reverse):
        mcts = PVMCTS(self.network, 0.35, epsilon=0, num_sims=20, env=env)
        return mcts.get_action_eval(state, action_mask, reverse)

# eval_func = get_eval_func(env, agents=[RandomAgent()], eval_n=20)

# score = eval_func(NewAgent(pv, env))
compare_agents(env, [
    RandomAgent(),
    MiniMaxAgent(1),
    MiniMaxAgent(2),
    MiniMaxAgent(3),
    MiniMaxAgent(4),
], 10)
print(score)
exit(0)

while True:
    print(env.render())
    _ = network.analyze_and_action(state, action_mask, False)
    action = int(input())

    next_state, reward, done, _ = env.step(action)
    state = next_state
    
    if done:
        print(env.render())
        break
