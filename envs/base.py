import gym
import numpy as np
from typing import Tuple, List, Callable, Any, Dict
from collections import namedtuple
import torch
from tqdm import tqdm
import abc

Transition = namedtuple(
    "Transition",
    ["state", "next_state", "action", "reward", "done", "action_mask", "next_action_mask", "black"]
)

class MCTSAbleEnv(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def hash(self) -> str:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def next(self, state:np.ndarray, action:int, current_player:int) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def isdone(self, state:np.ndarray, current_player:int) -> bool:
        raise NotImplementedError()
        
    @abc.abstractmethod
    def result(self, state:np.ndarray) -> Tuple[int, int]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_valid_action(self, state:np.ndarray, currnet_player:int) -> List[int]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def current_player(self, state:np.ndarray) -> int:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_random_action(self):
        raise NotImplementedError()


# random actionやaction_maskなどを要求する環境
# stepを行うごとにplayerが入れ替わるシステム
class BaseBoardEnv(gym.Env, metaclass=abc.ABCMeta):
    def __init__(self):
        super(BaseBoardEnv, self).__init__()
        self.player = 0
        self.step = self.step_dec(self.step)

    def step_dec(self, f):
        def _wrapper(*arg, **keywords):
            result = f(*arg, **keywords)
            self.player = 1 - self.player
            return result
        return _wrapper
    
    @abc.abstractmethod
    def render(self) -> None:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def reset(self) -> Tuple[np.ndarray, str, Dict[str, np.ndarray]]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        """
        output:
            Tuple(
                observation: np.ndarray[observation_space]
                reward: float
                done: bool
                {
                    "action_mask": np.ndarray[action_space]
                }
            )
        """
        raise NotImplementedError()
    
    @property
    @abc.abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def action_space(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_random_action(self) -> int:
        raise NotImplementedError()

class Agent:
    def __init__(self):
        pass

    def get_action_eval(self, env: BaseBoardEnv, state: np.ndarray, action_mask: np.ndarray) -> int:
        raise NotImplementedError()
    
    def get_action_expl(self, env: BaseBoardEnv, state: np.ndarray, action_mask: np.ndarray) -> int:
        raise NotImplementedError()

class RandomAgent(Agent):
    def __init__(self):
        self.name = "RandomAgent"
        self.get_action_eval = self.get_action
        self.get_action_expl = self.get_action
    
    def get_action(self, env: BaseBoardEnv, _hoge: np.ndarray, _fuga: np.ndarray) -> int:
        return env.get_random_action()

class ModelAgent(Agent):
    def __init__(self, model):
        self.name = "ModelAgent"
        self.model = model
    
    def get_action_eval(self, env: BaseBoardEnv, state: np.ndarray, action_mask: np.ndarray) -> int:
        state = torch.from_numpy(state).float()
        action_mask = torch.from_numpy(action_mask).float()
        action = self.model.get_action_eval(
            state,
            action_mask,
            env.player is not 0
        )
        return action
    
    def get_action_expl(self, env: BaseBoardEnv, state: np.ndarray, action_mask: np.ndarray) -> int:
        state = torch.from_numpy(state).float()
        action_mask = torch.from_numpy(action_mask).float()
        action = self.model.get_action_expl(
            state,
            action_mask,
            env.player is not 0
        )
        return action


def play(env:BaseBoardEnv , agent1: Agent, agent2: Agent, is_eval: bool) -> Tuple[bool, bool]:
    obs, info = env.reset()
    action_mask = info["action_mask"]
    if is_eval:
        agent1_action = agent1.get_action_eval
        agent2_action = agent2.get_action_eval
    else:
        agent1_action = agent1.get_action_expl
        agent2_action = agent2.get_action_expl
    
    while True:
        if env.player == 0:
            action = agent1_action(env, obs, action_mask)
        else:
            action = agent2_action(env, obs, action_mask)
        obs, reward, done, info = env.step(action)
        action_mask = info["action_mask"]
        if done:
            if reward == 1:
                return (True, False)
            elif reward == -1:
                return (False, True)
            else:
                return (False, False)

def get_explore_func(env:BaseBoardEnv) -> Callable[[Any], List[Transition]]:

    def explore_func(model) -> List[Transition]:
        agent = ModelAgent(model)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        data = []
        while True:
            action = agent.get_action_expl(env, obs, action_mask)
            black = 1 if env.player == 0 else -1
            next_obs, reward, done, info = env.step(action)
            next_action_mask = info["action_mask"]
            transition = Transition(
                state=obs,
                next_state=next_obs,
                action=action,
                reward=reward,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
                black=black
            )
            obs = next_obs
            action_mask = next_action_mask
            data.append(transition)
            if done:
                break
        return data
    return explore_func


def get_eval_func(env:BaseBoardEnv, agents: List[Agent], eval_n: int) -> Callable[[Any], List[Transition]]:
    def eval_func(model) -> Dict[str, float]:
        model_agent = ModelAgent(model)
        score = {}
        for agent in agents:
            win_black = 0
            win_white = 0
            draw_black = 0
            draw_white = 0
            for i in tqdm(range(eval_n // 2), leave=False, desc=f"[eval black: vs{agent.name}]"):
                result = play(env, model_agent, agent, True)
                if result[0]:
                    win_black += 1
                elif not result[1]:
                    draw_black += 1
            for i in tqdm(range(eval_n // 2), leave=False, desc=f"[eval white: vs{agent.name}]"):
                result = play(env, agent, model_agent, True)
                if result[1]:
                    win_white += 1
                elif not result[0]:
                    draw_white += 1
            score[f"eval/{agent.name}/win-black"] = win_black / (eval_n // 2)
            score[f"eval/{agent.name}/win-white"] = win_white / (eval_n // 2)
            score[f"eval/{agent.name}/draw-black"] = draw_black / (eval_n // 2)
            score[f"eval/{agent.name}/draw-white"] = draw_white / (eval_n // 2)
            score[f"eval/{agent.name}/score"] = (win_black + win_white + draw_black / 2 + draw_white / 2) / eval_n
        return score
    return eval_func