import gym
import numpy as np
from .base import BaseBoardEnv
from typing import Tuple, Dict, List
import random

HUSH_ARRAY = np.array([
    1,2,4,
    8,16,32,
    64,128,0
])

def check(b: np.ndarray) -> bool:
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

class OXEnv(BaseBoardEnv):
    action_num = 9
    def __init__(self):
        super(OXEnv, self).__init__()
        self.black_board = np.zeros(9, dtype=np.uint8)
        self.white_board = np.zeros(9, dtype=np.uint8)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.black_board = np.zeros(9, dtype=np.uint8)
        self.white_board = np.zeros(9, dtype=np.uint8)
        return np.zeros(18, dtype=np.uint8), {"action_mask": np.zeros(9, dtype=np.uint8)} 
    
    def render(self):
        s = ""
        for i in range(3):
            for j in range(3):
                idx = 3 * i + j
                if self.black_board[idx] == 1:
                    s += "O"
                elif self.white_board[idx] == 1:
                    s += "X"
                else:
                    s += "-"

            s += "\n"
        print(s)
    
    @property
    def observation_space(self):
        return gym.spaces.MultiBinary(18)
    
    @property
    def action_space(self):
        return gym.spaces.Discrete(9)
    
    def put(self, action: int):
        if self.player == 0:
            self.black_board[action] = 1
        else:
            self.white_board[action] = 1
        return 
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        self.put(action)
        reward = 0
        if self.player == 0:
            done = check(self.black_board)
            if done:
                reward = 1
        else:
            done = check(self.white_board)
            if done:
                reward = -1
        stone = self.black_board + self.white_board
        done = done or np.sum(stone) == 9

        action_mask = 1 - stone
        return np.stack([self.black_board, self.white_board]).reshape(-1), reward, done, {"action_mask": action_mask}

    def init(self) :
        return np.zeros(18, dtype=np.uint8)
    
    def hash(self, state):
        compressed_a = state[0:9] * HUSH_ARRAY
        compressed_b = state[9:18] * HUSH_ARRAY

        s = ""
        s += chr(np.sum(compressed_a[:-1])) + str(state[8])
        s += chr(np.sum(compressed_b[:-1])) + str(state[17])
        return s
    
    def next(self, state, action, current_player) -> Tuple[np.ndarray, bool]:
        next_state = state.copy()
        if current_player == 0:
            next_state[action] = 1
        else:
            next_state[action + 9] = 1
        return next_state, self.isdone(next_state, 1 - current_player)
    
    def isdone(self, state, current_player) -> bool:
        return np.sum(state) == 9 or check(state[0:9]) or check(state[9:18])

    def result(self, state) -> Tuple[int, int]:
        if check(state[0: 9]):
            return 1, -1
        if check(state[9:18]):
            return -1, 1
        return 0, 0
    
    def get_valid_action(self, state, current_player) -> List[int]:
        actions = []
        stone = state[0: 9] + state[9: 18]
        for i, v in enumerate(stone):
            if v == 0:
                actions.append(i)
        return actions
    
    def current_player(self, state) -> int:
        return np.sum(state) % 2
    
    def get_random_action(self):
        board = np.concatenate([self.black_board, self.white_board], axis=0)
        actions = self.get_valid_action(board, self.player)
        return random.choice(actions)
        
        