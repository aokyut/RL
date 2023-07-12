import gym
import numpy as np
import random
import torch
from tqdm import tqdm


def check(b):
    return (
        b[0] != 0 and (b[0] == b[1] and b[1] == b[2] or
        b[0] == b[3] and b[3] == b[6] or
        b[0] == b[4] and b[4] == b[8]) or
        b[1] != 0 and b[1] == b[4] and b[4] == b[7] or
        b[2] != 0 and b[2] == b[5] and b[5] == b[8] or
        b[2] != 0 and b[2] == b[4] and b[4] == b[6] or
        b[3] != 0 and b[3] == b[4] and b[4] == b[5] or
        b[6] != 0 and b[6] == b[7] and b[7] == b[8]
    )

class OXEnv(gym.Env):
    def __init__(self):
        super(OXEnv, self).__init__()
        self.board = [0] * 9
        self.player = 1
    
    def action_mask(self):
        return (np.array(self.board) == 0).astype(np.int)
    
    def reset(self):
        self.board = [0] * 9
        self.player = 1
        b = (np.array(self.board) == 1).astype(np.int)
        w = (np.array(self.board) == 2).astype(np.int)

        return np.concatenate((b, w)).reshape(-1)

    def step(self, action):
        self.board[action] = self.player
        done = sum(self.board) == 13
        reward = 0
        if check(self.board):
            done = True
            if self.player == 1:
                reward = 1
            else:
                reward = -1
        self.player = 3 - self.player
        b = (np.array(self.board) == 1).astype(np.int)
        w = (np.array(self.board) == 2).astype(np.int)

        return np.concatenate((b, w)).reshape(-1), reward, done, {}
    
    def render(self):
        s = "\n"
        for i in range(3):
            for j in range(3):
                if self.board[i * 3 + j] == 1:
                    s += "O"
                elif self.board[i * 3 + j] == 2:
                    s += "X"
                else:
                    s += "-"
            s += "\n"
        return s
    def out(self):
        s = self.render()
        with open("out", "a") as f:
            f.write(s)

class RandomAgent:
    name = "RandomAgent"
    def action(self, b, obs, action_mask):
        return random.choice(np.where(action_mask > 0)[0])

class ModelAgent:
    name = "ModelAgent"
    def __init__(self, network):
        self.model = network
    def action(self, b, obs, action_mask):
        obs = torch.from_numpy(obs).float()
        action_mask = torch.from_numpy(action_mask).float()
        action = self.model.get_action_eval(obs, action_mask, not b.player == 1)
        return action

def play(agent1, agent2, render = False):
    b = OXEnv()
    agents = [agent1, agent2]
    player = 0
    obs = b.reset()
    action_mask = b.action_mask()
    while True:
        # b.render()
        if render:
            b.out()
        agent = agents[player]
        action = agent.action(b, obs, action_mask)
        obs, reward, done, _ = b.step(action)
        action_mask = b.action_mask()
        player = 1 - player
        if done:
            # b.render()
            if reward > 0: 
                return [True, False]
            elif reward < 0: 
                return [False, True]
            else: 
                return[False, False]

def eval_func(model):
    """
    input: model
    output: dict{key, score}
        key: str
        score: float
    """
    tar_agent = ModelAgent(model)
    agents = [
        RandomAgent()
    ]
    score = {}
    n = 50
    for agent in tqdm(agents, leave=False):
        w_black = 0
        w_white = 0
        for _ in tqdm(range(n // 2), leave=False, desc=agent.name):
            if play(tar_agent, agent)[0]:
                w_black += 1
            if play(agent, tar_agent)[1]:
                w_white += 1
        score[f"eval/{agent.name}-mean"] = (w_black + w_white) / n
        score[f"eval/{agent.name}-black"] = w_black / (n // 2)
        score[f"eval/{agent.name}-white"] = w_white / (n // 2)
    play(tar_agent, agent, True)
    play(agent, tar_agent, True)
    # w =0
    # for i in range(n):
    #     if random.random() < 0.5:
    #         w+=1
    # score["eval/random_seq"] = w/ n
    return score



def explore_func(model):
    """
    input: model
    output: data{state, next_state, action, done, reward}
    """
    env = OXEnv()
    state = env.reset()
    data = []
    while True:
        action_mask = env.action_mask()
        black =1 if env.player == 1 else -1
        action = model.get_action(torch.from_numpy(state).float(), torch.from_numpy(action_mask).float(), not env.player == 1)
        next_state, reward, done, _ = env.step(action)
        next_action_mask = env.action_mask()
        # 学習には次の状態のaction_maskのみを使うから、⬆️のようにaction_maskを呼び出す
        data.append(
            {"state": state,
            "next_state": next_state,
            "action": action,
            "action_mask": action_mask,
            "next_action_mask": next_action_mask,
            "done": np.array([1.0]) if done else np.array([0.0]),
            "reward": np.array([reward]),
            "black": np.array([black])}
        )
        state = next_state
        if done:
            break
    
    return data