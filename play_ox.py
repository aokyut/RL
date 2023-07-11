from gomoku_env.test_env import OXEnv, eval_func
from network import Dense, DenseConfig
import torch

net_config = DenseConfig.parse()
network = Dense(net_config)

network.load_state_dict(torch.load("checkpoint/ox_test/10000.pth"))

env = OXEnv()

state = env.reset()

score = eval_func(network)
print(score)

while True:
    action_mask = env.action_mask()
    print(env.render())
    action = network.get_action_eval(torch.from_numpy(state).float(), torch.from_numpy(action_mask).float(), not env.player == 1)
    action_val = {}
    vals = network(torch.from_numpy(state).float()).detach().numpy()
    for i in range(9):
        if action_mask[i] == 0:
            continue
        action_val[i] = vals[i]

    print(action_val)
    next_state, reward, done, _ = env.step(action)
    
    if done:
        print(env.render())
        break
