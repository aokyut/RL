import gym
import numpy as np
import random
from .base import BaseBoardEnv
from typing import Tuple, Dict, List

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
    action_num=16
    def __init__(self):
        super(Qubic, self).__init__()
        self.black_board = np.zeros(64, dtype=np.uint8)
        self.white_board = np.zeros(64, dtype=np.uint8)
        self.stone_board = np.zeros(64, dtype=np.uint8)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.black_board = np.zeros(64, dtype=np.uint8)
        self.white_board = np.zeros(64, dtype=np.uint8)
        self.stone_board = np.zeros(64, dtype=np.uint8)
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

    def init(self):
        return np.zeros(128, dtype=np.uint8)

    def hash(self, state):
        compressed_board = state[0:16] +\
                           2 * state[16:32] + \
                           4 * state[32:48] + \
                           8 * state[48:64] + \
                           16 * state[64:80] + \
                           32 * state[80:96] + \
                           64 * state[96:112] + \
                           128 * state[112:128]
        s = ""
        for num in compressed_board:
            s += chr(num)
        return s
    
    def next(self, state, action, current_player) -> Tuple[np.ndarray, bool]:
        """
        output:
            next_state
            done
        """
        stone = state[0: 64] + state[64: 128]
        if stone[action] == 0:
            pass
        elif stone[action + 16] == 0:
            action += 16
        elif stone[action + 32] == 0:
            action += 32
        else:
            if stone[action + 48] == 1:
                raise NotImplementedError()
            action += 48
        if current_player == 1:
            action += 64
        new_state = state.copy()
        new_state[action] = 1
        return new_state, self.isdone(new_state, 1 - current_player)

    def isdone(self, state, current_player) -> bool:
        return np.sum(state) ==  64 or check(state[0:64]) or check(state[64:128])

    def result(self, state) -> Tuple[int, int]:
        if check(state[0:64]):
            return 1, -1
        if check(state[64:128]):
            return -1, 1
        return 0, 0

    def get_valid_action(self, state, current_player) -> List[int]:
        actions = []
        stone = state[0:64] + state[64: 128]
        for i, v in enumerate(stone[48:64]):
            if v == 0:
                actions.append(i)
        return actions

    def current_player(self, state) -> int:
        return np.sum(state) % 2

    def get_random_action(self):
        board = np.concatenate([self.black_board, self.white_board], axis=0)
        actions = self.get_valid_action(board, self.player)
        return random.choice(actions)