import numpy as np


class Renju:
    def __init__(self):
        pass

    def reset(self):
        self.board = np.zeros((2, 15, 15))
        self.piece = 0
        self.action_mask = np.zeros((2, 15, 15))
        self.action_mask[self.piece, 7, 7] = 1
        self.mode = 1
        self.done = False

        return self.action_mask, self.board, self.piece, self.done

    def step(self, action):
        """
        action: np.array(2) [h, w]
        """
        self.board[self.piece, action[0], action[1]] = 1

        if self.check(action):
            print("end")
            self.done = True
            return self.action_mask, self.board, self.piece, self.done
        if self.board.sum() > 99:
            return self.action_mask, self.board, self.piece, self.done
        self.piece = 1 - self.piece

        if self.mode > 5:
            self.action_mask = np.zeros((2, 15, 15))
            self.action_mask[self.piece, :, :] = 1
        else:
            self.action_mask = np.zeros((2, 15, 15))
            self.action_mask[self.piece, 7 - self.mode: 8 + self.mode, 7 - self.mode: 8 + self.mode] = 1
        self.mode += 1

        self.action_mask = np.maximum(self.action_mask - np.amax(self.board, axis=0, keepdims=True), 0)

        return self.action_mask, self.board, self.piece, self.done

    def check(self, action):
        """
        action:  [h, w]
        """
        vecs = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
        pos = action

        for vec in vecs:
            count = 0
            pos_ = pos
            while True:
                pos_ = pos_ + vec
                if pos_.max() > 14 or pos_.min() < 0:
                    break
                if self.board[self.piece, pos_[0], pos_[1]] == 1:
                    count += 1
                else:
                    break
            vec *= -1
            pos_ = pos
            while True:
                pos_ = pos_ + vec
                if pos_.max() > 14 or pos_.min() < 0:
                    break
                if self.board[self.piece, pos_[0], pos_[1]] == 1:
                    count += 1
                else:
                    break
            if count > 3:
                return True

        return False

    def show(self):
        s = ""
        for i in range(15):
            for j in range(15):
                if self.board[0, i, j] == 1:
                    s += "o"
                elif self.board[1, i, j] == 1:
                    s += "x"
                elif self.action_mask.sum(axis=0)[i, j] == 1:
                    s += "-"
                else:
                    s += "#"
            s += "\n"
        print(s)

    def __str__(self):
        s = ""
        for i in range(15):
            for j in range(15):
                if self.board[0, i, j] == 1:
                    s += "o"
                elif self.board[1, i, j] == 1:
                    s += "x"
                else:
                    s += "-"
        return s

    def mstep(self):
        renju = Renju()
        renju.reset()


def b2s(board):
    s = ""
    for i in range(15):
        for j in range(15):
            if board[0, i, j] == 1:
                s += "o"
            elif board[1, i, j] == 1:
                s += "x"
            else:
                s += "-"
    return s


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
    piece = board.sum() % 2
    if board.sum() > 5:
        action_mask = np.zeros((2, 15, 15))
        action_mask[piece, :, :] = 1
    else:
        mode = board.sum()
        action_mask = np.zeros((2, 15, 15))
        action_mask[piece, 7 - mode: 8 + mode, 7 - mode: 8 + mode] - 1
    return np.where(action_mask > 0)[0]


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
    piece = board.sum() % 2
    if board.sum() > 5:
        action_mask = np.zeros((2, 15, 15))
        action_mask[piece, :, :] = 1
    else:
        mode = board.sum()
        action_mask = np.zeros((2, 15, 15))
        action_mask[piece, 7 - mode: 8 + mode, 7 - mode: 8 + mode] - 1
    return action_mask


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
    piece = board.sum() % 2
    assert action >= 0 and action <= 449, f"action: {action}"
    assert action // 225 == piece, f"action: {action}, piece: {piece}"
    next_board = board.copy()
    action = action % 225
    i, j = action // 15, action % 15
    next_board[piece, i, j] = 1
    if check(board, [i, j], piece):
        return (next_board, True, True)
    if next_board.sum() > 99:
        return (next_board, True, False)
    return (next_board, False, False)


def check(board, action, piece):
    """
    action:  [h, w]
    """
    vecs = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
    pos = action

    for vec in vecs:
        count = 0
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() > 14 or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        vec *= -1
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() > 14 or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        if count > 3:
            return True

    return False


def get_initial_board():
    # mask = np.zeros((2, 15, 15))
    # mask[0, 7, 7] = 1
    return np.zeros((2, 15, 15))
