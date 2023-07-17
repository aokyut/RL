from envs.ox_alpha import OXEnv
from envs.base import get_eval_func, RandomAgent
from network import PVNet, ResNet, BottleNeckBlock, Policy2d, Value2d
from alphazero import PVMCTS
import torch


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

pv.load_state_dict(torch.load("checkpoint/alphazero_ox_debug5e/latest.pth"))

env = OXEnv()

network = PVMCTS(pv, 0.35, epsilon=0.25, num_sims=100, env=env)

# score = eval_func(NewAgent(pv, env))
# print(score)
# exit(0)

for i in range(10):

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

    print(env.result(state))