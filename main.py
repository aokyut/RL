from network import PVNet, Policy2d, Value2d, ResNet, BottleNeckBlock
from alphazero import AlphaZero, AlphaZeroConfig
from envs.qubic import Qubic, show
from envs.base import get_eval_func
from PyBGEnv import qubic
from tqdm import tqdm
import random
from argparse import ArgumentParser
import os
import torch
from utills import parse_from_dataclass

input_net = ResNet(
    in_ch=8, out_ch=32, 
    hidden_ch=128, block_n=4, block_type=BottleNeckBlock,
)

policy_net = Policy2d(
    in_ch=32,
    out_ch=8,
    in_fc=128,
    out_fc=16
)

value_net = Value2d(
    in_ch=32,
    out_ch=8,
    in_fc=128
)

pv = PVNet(input_layer=input_net, policy_layer=policy_net, value_layer=value_net, in_shape=[-1, 8, 4, 4])

env = qubic
# eval_func = get_eval_func(env, [RandomAgent(), MiniMaxAgent(1)], 20)


def play(agent1, agent2):
    state = env.init()
    player = 0
    t = 0
    agents = [agent1, agent2]
    while True:
        t += 1
        action_mask = env.get_action_mask(state)
        agent = agents[player]
        action = agent.get_action(env, state, action_mask)
        next_state = env.get_next(state, action, player)
        state = next_state
        res = env.is_win(next_state, player)
        if res:
            if player == 0:
                return 1
            else:
                return -1
        if env.is_draw(next_state):
            return 0
        player = 1 - player

class RandomAgent:
    name = "Random"
    def get_action(self, env, state, action_mask):
        valid_action = env.valid_actions(state, env.current_player(state))
        return random.choice(valid_action)

class MiniMaxAgent:
    def __init__(self, depth):
        self.depth = depth
        self.name = "MiniMax" + str(depth)
    
    def get_action(self, env, state, action_mask):
        return env.minimax_action(state, env.current_player(state), self.depth)

class ModelAgent:
    def __init__(self, model):
        self.model = model
        self.name = "Model"
    
    def get_action(self, env, state, action_mask):
        return self.model.get_action_eval(state, action_mask, False)

def eval_func(model):
    eval_n = 20
    tar_agent = ModelAgent(model)
    agents = [
        RandomAgent(),
        MiniMaxAgent(1),
    ]
    record = {}
    for agent in agents:
        win_black = 0
        win_white = 0
        draw_black = 0
        draw_white = 0
        score = 0
        for i in tqdm(range(eval_n // 2), leave=False, desc=f"[{agent.name}-black]"):
            result = play(tar_agent, agent)
            if result == 1:
                win_black += 1
            elif result == 0:
                draw_black += 1
        for i in tqdm(range(eval_n // 2), leave=False, desc=f"[{agent.name}-white]"):
            result = play(agent, tar_agent)
            if result == -1:
                win_white += 1
            elif result == 0:
                draw_white += 1
        score = (win_black + win_white + draw_black / 2 + draw_white / 2) / eval_n
        record[f"vs{agent.name}/win_black"] = 2 * win_black / eval_n
        record[f"vs{agent.name}/win_white"] = 2 * win_white / eval_n
        record[f"vs{agent.name}/draw_black"] = 2 * draw_black / eval_n
        record[f"vs{agent.name}/draw_white"] = 2 * draw_white / eval_n
        record[f"vs{agent.name}/score"] = score
    return record


config = AlphaZeroConfig(
    buffer_size=40000,
    log_name="alphazero_qubic",
    prob_action_th=4,
    selfplay_n=200,
    episode=200,
    batch_size=128,
    epoch=10,
    sim_n=50,
    save_dir="checkpoint"
)

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--save_dir", default="checkpoint", type=str)
    # parser.add_argument("--load", action="store_true", default=False)
    # parser.add_argument("--use_ray", action="store_true", default=False)
    # parser.add_argument("--load_path", type=str, default="hoge")
    # parser.add_argument("--buffer_size", default=40000, type=int)
    # parser.add_argument("--epoch", default=10, type=int)
    # parser.add_argument("--episode", default=200, type=int)
    # args = parser.parse_args()

    # config.save_dir = args.save_dir
    # config.buffer_size = args.buffer_size
    # config.epoch = args.epoch
    # config.episode = args.episode
    # config.use_ray = args.use_ray

    config = parse_from_dataclass(AlphaZeroConfig)

    if config.load:
        path = config.load_path
        assert os.path.exists(path)
        pv.load_state_dict(torch.load(path))
            

    alphazero  = AlphaZero(pv, config, env, eval_func)

    alphazero.train()