import gym
import numpy as np
import random
from .base import BaseBoardEnv
from typing import Tuple, Dict

def check(board: np.ndarray) -> bool:
    """
    board_array: np.ndarray[64]
    """
    if np.sum(board[0] * board[21] * board[42] * board[63]) > 0:
        return True
    board = board.reshape([4, 4, 4,])
    if np.sum(np.prod(board, axis=0)) > 0:
        return True
    if np.sum(np.prod(board, axis=1)) > 0:
        return True
    if np.sum(np.prod(board, axis=2)) > 0:
        return True
    
    return False

def show(board_array: np.ndarray):
    print(board_array)

class Qubic(BaseBoardEnv):
    def __init__(self):
        super(Qubic, self).__init__()
        self.black_board = np.zeros(64, dtype=np.uint8)
        self.white_board = np.zeros(64, dtype=np.uint8)
        self.stone_board = np.zeros(64, dtype=np.uint8)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return np.zeros(128, dtype=np.uint8), {"action_mask": np.zeros(16, dtype=np.uint8)}

    def render(self):
        s = ""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    idx = 4 * i + 16 * j + k
                    if self.black_board[idx] == 1:
                        s += "O"
                    elif self.white_board[idx] == 1:
                        s += "X"
                    else:
                        s += "-"
                if j == 3: break
                s += "＿/￣"
            s += "\n"
        print(s)

    @property
    def observation_space(self):
        return gym.spaces.MultiBinary(64)

    @property
    def action_space(self):
        return gym.spaces.Discrete(16)
    
    def put(self, action):
        if self.player == 0:
            self.black_board[action] = 1
        else:
            self.white_board[action] = 1
        self.stone_board[action] = 1
        

    def step(self, action):
        action %= 16
        if self.stone_board[action] == 0:
            action = action
        elif self.stone_board[action + 16] == 0:
            action = action + 16
        elif self.stone_board[action + 32] == 0:
            action = action + 32
        else:
            action = action + 48
        
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
        
        done = done or np.sum(self.stone_board) ==  64

        action_mask = self.stone_board[48:]
        return np.stack([self.black_board, self.white_board]).reshape(-1), reward, done, {"action_mask": action_mask}
