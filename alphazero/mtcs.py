from renju import b2s, get_action_mask, get_next_state, get_valid_actions, BOARD_SIZE
import numpy as np
import math
import random
import renju
from tqdm import tqdm


def softmax_select(p, t=0.5):
    """
    p: np.Array(n)
    t: float
    """
    u = np.sum(np.exp(p * t))
    return np.exp(p * t) / u


class PlayerAgent:
    name = "Player"

    def action(self, state):
        action_mask = np.sum(get_action_mask(state), axis=0)
        valid_action = get_valid_actions(state)
        s = ""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[0, i, j] == 1:
                    s += "O"
                elif state[1, i, j] == 1:
                    s += "X"
                elif action_mask[i, j] == 1:
                    s += "#"
                else:
                    s += "-"
            s += "\n"
        print(s)
        while True:
            i, j = tuple(map(int, input(">>").strip().split()))
            action = i * BOARD_SIZE + j + BOARD_SIZE * BOARD_SIZE * (state.sum() % 2)
            if action in valid_action:
                return int(action)
            else:
                print("valid_action", list(valid_action))


class RandomAgent:
    name = "Random"

    def action(self, state):
        actions = get_valid_actions(state)
        return random.choice(actions)


class UCTAgent:
    name = "UCT"

    def __init__(self, num_sims, c_uct=1):
        self.num_sims = num_sims
        self.c_uct = c_uct

    def action(self, state):
        valid_actions = get_valid_actions(state)
        N = {}
        W = {}
        next_states = {}
        for action in valid_actions:
            next_state, done, iswin = get_next_state(state, action)
            if done:
                if iswin:
                    v = 1.0
                else:
                    v = 0.0
            else:
                v = self.playout(next_state)
            N[action] = 1.0
            W[action] = v
            next_states[action] = next_state

        for _ in tqdm(range(self.num_sims), leave=False):
            sum_n = sum(N.values())
            p = [((W[action] / N[action]) + self.c_uct * math.sqrt(2.0 * math.log(sum_n) / N[action]), action) for action in valid_actions]
            _, action = max(p, key=lambda x: x[0])
            v = self.playout(next_states[action])
            N[action] += 1.0
            W[action] += v
        p = [(N[action], action) for action in valid_actions]
        _, action = max(p, key=lambda x: x[0])
        # p = sorted(p, key=lambda x: x[0])
        # for value, action in p:
        #     print((action % (BOARD_SIZE * BOARD_SIZE)) // BOARD_SIZE, action % BOARD_SIZE, value)

        return action

    def playout(self, state):
        coef = -1.0
        while True:
            actions = get_valid_actions(state)
            action = random.choice(actions)
            next_state, isdone, iswin = get_next_state(state, action)
            if isdone:
                if iswin:
                    return 1.0 * coef
                else:
                    return 0.0
            coef *= -1.0
            state = next_state


class PUCTAgent:
    name = "PUCT"

    def __init__(self, num_sims):
        self.N = {}
        self.W = {}
        self.next_states = {}

    def search(self, root_state):
        pass

    def eval(self, state):
        pass

    def expand(self, state):
        pass


class AlphaZeroAgent:
    name = "AlphaZero"

    def __init__(self, network, num_sims):
        self.mtcs = PVmtcs(network, 1, epsilon=0.0)
        self.num_sims = num_sims

    def action(self, state):
        """
        Parameters
        -----
        state: numpy.array [2, 15, 15]

        Returns
        -----
        action: int 0~ACTION_SPACE
        """
        mtcs_policy = self.mtcs.search(state, self.num_sims)
        action = random.choice(np.where(np.array(mtcs_policy) == max(mtcs_policy))[0])
        return action


class PVmtcs:
    def __init__(self, network, alpha, c_puct=1.0, epsilon=0.25):
        self.network = network
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon

        self.P = {}
        self.N = {}
        self.W = {}
        self.Done = {}
        self.IsWin = {}

        self.next_states = {}
        self.network.eval()

    def search(self, root_state, num_sims):
        s = b2s(root_state)

        if s not in self.P:
            _ = self.expand(root_state)

        valid_actions = get_valid_actions(root_state)

        #: root状態にだけは事前確立にディリクレノイズをのせて探索を促進する
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha] * len(valid_actions))
        for a, noise in zip(valid_actions, dirichlet_noise):
            self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        #: MCTS simulationの実行
        for _ in tqdm(range(num_sims), leave=False):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(renju.ACTION_SPACE)]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]

            # print(self.P[s], self.N[s], self.W[s])

            #: PUCTスコアの算出
            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])
            prob = softmax_select(scores)
            #: スコアのもっとも高いactionを選択
            if len(scores) == 0:
                print(self.P[s], self.U, self.P, self.next_states[s])
            action = np.random.choice(list(range(renju.ACTION_SPACE)), p=prob)
            next_state = self.next_states[s][action]

            #: 選択した行動を評価（次は相手番なので評価値に-1をかける）
            v = -self.evaluate(next_state)

            self.W[s][action] += v
            self.N[s][action] += 1

        #: mcts_policyは全試行回数に占める各アクションの試行回数の割合
        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def expand(self, state):
        s = b2s(state)
        action_mask = get_action_mask(state).ravel()
        policy, value = self.network.predict(state, action_mask)
        self.P[s] = policy.detach().numpy()
        self.N[s] = [0] * renju.ACTION_SPACE
        self.W[s] = [0] * renju.ACTION_SPACE

        actions = np.where(action_mask > 0)[0]
        next_states = {}
        for action in actions:
            next_state, done, win = get_next_state(state, action)
            next_states[action] = next_state
            _s = b2s(next_state)
            self.Done[_s] = done
            self.IsWin[_s] = win
        self.next_states[s] = next_states
        # print(self.next_states)
        # print(value.shape)
        return value.detach().numpy()[0]

    def evaluate(self, state):
        s = b2s(state)

        if self.Done[s]:
            if self.IsWin[s]:
                return 1.0
            return 0
        elif s not in self.P:
            value = self.expand(state)
            return value
        else:
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(renju.ACTION_SPACE)]
            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]
            valid_actions = get_valid_actions(state)
            scores = [u + q for u, q in zip(U, Q)]

            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            best_action = random.choice(np.where(scores == scores.max())[0])

            next_state = self.next_states[s][best_action]

            v = -self.evaluate(next_state)

            self.W[s][best_action] += v
            self.N[s][best_action] += 1

            return v
