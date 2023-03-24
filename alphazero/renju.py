import numpy as np
from tqdm import tqdm
import tkinter as tk
from numba import jit
import codecs

BOARD_SIZE = 7
ACTION_SPACE = BOARD_SIZE * BOARD_SIZE * 2
C = BOARD_SIZE // 2
BOARD_TH = min(BOARD_SIZE // 2, 5)
END_TH = BOARD_SIZE * BOARD_SIZE // 2
INPUT_CH = 2

WINDOW_SIZE = 400
BOARD_COLOR = "green"
FIRST_STONE = "black"
SECOND_STONE = "white"
PLAYABLE_COLOR = "yellow"


@jit("f8[:,:,:]()", nopython=True, cache=True)
def get_init():
    return np.zeros((2, BOARD_SIZE, BOARD_SIZE))


@jit("string(f8[:,:,:])", nopython=True, cache=True)
def b2s(board):
    s = ""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[0, i, j] == 1:
                s += "o"
            elif board[1, i, j] == 1:
                s += "x"
            else:
                s += "-"
    return s


@jit("f8[:,:,:](f8[:,:,:])", nopython=True, cache=True)
def get_action_mask(board):
    """
    Paramters:
    -----
    board: numpy.array(2, 15, 15)

    Returns:
    -----
    numpy.array(2, 15, 15)
        action_mask
    """
    piece = int(board.sum() % 2)
    if board.sum() > BOARD_TH:
        action_mask = np.zeros((2, BOARD_SIZE, BOARD_SIZE))
        action_mask[piece, :, :] = 1
    else:
        mode = int(board.sum())
        action_mask = np.zeros((2, BOARD_SIZE, BOARD_SIZE))
        # print(C, mode, piece)
        action_mask[piece, C - mode: C + 1 + mode, C - mode: C + 1 + mode] = 1
    stone_mask = np.sum(board, axis=0).reshape(BOARD_SIZE, BOARD_SIZE)
    action_mask[0, :, :] -= stone_mask
    action_mask[1, :, :] -= stone_mask

    cliped = np.clip(action_mask, 0, 1)

    # action_mask = np.clip(action_mask, 0, 1)
    # print(action_mask)
    return cliped


@jit("i8[:](f8[:,:,:])", nopython=True, cache=True)
def get_valid_actions(board):
    """
    Parameters:
    -----
    board: numpy.array(2, 15, 15)

    Returns:
    ------
    numpy.array(*,)
        valid_actions
    """
    action_mask = get_action_mask(board)
    return np.where(action_mask.ravel() > 0)[0]


@jit("b1(f8[:,:,:], i8, i8, i8)", nopython=True, cache=True)
def check(board, piece, i, j):
    """
    Parameters
    -----
    board: numpy.array([2, 15, 15])
    piece: 0 or 1
    i: int 0~board_size
    j: int 0~board_size
    """
    vecs = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
    pos = np.int64([i, j])

    for vec in vecs:
        count = 0
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() >= BOARD_SIZE or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        vec *= -1
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() >= BOARD_SIZE or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        if count > 3:
            return True

    return False


@jit("Tuple((f8[:,:,:], b1, b1))(f8[:,:,:], i8)", nopython=True, cache=True)
def get_next_state(board, action):
    """
    Parameters
    -----
    board: numpy.array(2, 15, 15)
    action: int 0~449

    Returns
    -----
    (next_board, done, iswin)
    next_board: numpy.array(2, 15, 15)
    done: bool
    iswin: bool
    """
    piece = int(board.sum() % 2)
    # assert action >= 0 and action <= ACTION_SPACE - 1, f"action: {action}"
    # assert action // (BOARD_SIZE * BOARD_SIZE) == piece, f"action: {action}, piece: {piece}"
    next_board = board.copy()
    action = action % (BOARD_SIZE * BOARD_SIZE)
    i, j = action // BOARD_SIZE, action % BOARD_SIZE
    next_board[piece, i, j] = 1
    if check(board, piece, i, j):
        return (next_board, True, True)
    if next_board.sum() > END_TH:
        return (next_board, True, False)
    return (next_board, False, False)


@jit('string(f8[:, :, :])', nopython=True, cache=True)
def show_board(state):
    s = ""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state[0, i, j] == 1:
                s += "O"
            elif state[1, i, j] == 1:
                s += "X"
            else:
                s += "-"
        s += "\n"
    # print(s)
    return s


def eval_play(agent1, agent2, num):
    """
    Parameters
    -----
    agent1: Agent
    agent2: Agent
    num: int

    Returns
    -----
    p1_win_rate: float
    p2_win_rate: float
    """
    p1_win = 0
    p2_win = 0
    for i in tqdm(range(num // 2), leave=False, desc=f"[{agent1.name}] vs [{agent2.name}]"):
        p1, p2 = play(agent1, agent2)
        p1_win += p1
        p2_win += p2
        p2, p1 = play(agent2, agent1)
        p1_win += p1
        p2_win += p2
    print(p1_win, p2_win)
    return p1_win / num, p2_win / num


def play(agent1, agent2):
    state = get_init()
    done = False
    agents = [agent1, agent2]
    player = 0
    wins = [0, 0]
    while True:
        agent = agents[player]
        action = agent.action(state)
        next_state, done, iswin = get_next_state(state, action)
        if done:
            if iswin:
                wins[player] = 1
                # print(agent.name)
                # print(show_board(next_state))
                # print(wins)
                return tuple(wins)
            else:
                return tuple(wins)
        player = 1 - player
        state = next_state
    


class Game:
    def __init__(self, agent, first=False):
        self.player = 0
        if first:
            self.PLAYER = 0
        else:
            self.PLAYER = 1
        self.root = tk.Tk()
        self.root.title("Gomoku")
        self.root.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}")
        self.size = WINDOW_SIZE // BOARD_SIZE
        self.state = get_init()
        self.valid_actions = get_valid_actions(self.state)
        self.done = False
        self.agent = agent

        self.init_canvas()
        self.init_board()

        if not first:
            self.com_action()

        self.root.mainloop()

    def init_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            bg=BOARD_COLOR,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE,
            highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind('<ButtonPress>', self.click)
    
    def init_board(self):
        board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                xs = x * self.size
                ys = y * self.size
                xe = (x + 1) * self.size
                ye = (y + 1) * self.size

                tag_name = "square" + str(y) + str(x)
                self.canvas.create_rectangle(
                    xs,ys,
                    xe,ye,
                    tag=tag_name
                )
        
    def put(self, x, y):
        xs = x * self.size
        ys = y * self.size
        xe = (x + 1) * self.size
        ye = (y + 1) * self.size
        if self.player == 0:
            color = FIRST_STONE
        else:
            color = SECOND_STONE
        self.canvas.create_oval(
            xs, ys,
            xe, ye,
            fill=color
        )
        show_board(self.state)

    # クリックされた時のイベントの設定
    def click(self, event):
        print("click", self.player, self.PLAYER, self.done)
        if self.player != self.PLAYER or self.done:
            return
        print("flag", self.valid_actions)
        x = event.x // self.size
        y = event.y // self.size
        action = y * BOARD_SIZE + x + self.player * BOARD_SIZE * BOARD_SIZE
        if action in self.valid_actions:
            self.put(x, y)
            next_state, self.done, self.iswin = get_next_state(self.state, action)
            self.end_check()
            self.state = next_state
            self.valid_actions = get_valid_actions(self.state)
            self.player = 1 - self.player
            self.root.after(10, self.com_action)
    
    def com_action(self):
        if self.done:
            return
        action = self.agent.action(self.state) 
        self.put(action % BOARD_SIZE, (action % (BOARD_SIZE * BOARD_SIZE)) // BOARD_SIZE)
        next_state, self.done, self.isdone = get_next_state(self.state, action)
        self.end_check()
        self.player = 1 - self.player
        self.state = next_state
        self.valid_actions = get_valid_actions(self.state)
        print(self.player)
    
    def end_check(self):
        if self.done:
            print("end")
            if self.iswin:
                print(f"player-{self.player} win")
            else:
                print("draw")
