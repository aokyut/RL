from network import PVNet
from utills import ConfigParser, save_model
from dataclasses import dataclass
from memory import ReplayMemory
from os.path import join
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
from tqdm import tqdm
from typing import Dict, Any, List
import random

@dataclass
class AlphaZeroConfig:
    buffer_size: int = 2 << 15
    batch_size: int = 64
    log_name: str = "alphazero"
    log_n: int = 100
    eval_n: int = 1000
    save_n: int = 20000
    sim_n: int = 40
    prob_action_th: int = 2 #この手数に達するまでは探索回数に比例する確立で行動する
    dirichlet_alpha: float = 0.35
    epoch: int = 100
    selfplay_n: int = 300

class PVMCTS:
    def __init__(self, network, alpha, env, c_puct=1.0, epsilon=0.25, num_sims=50):
        self.network = network
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon

        self.hash = env.hash
        self.next = env.next
        self.is_done = env.isdone
        self.result = env.result
        self.get_valid_actions = env.get_valid_action
        self.action_num = env.action_num
        self.env = env

        self.num_sims = num_sims

        #: prior probability
        self.P = {}
        
        #: visit count
        self.N = {}
        
        #: Wは累計価値
        #: Q(s, a) = W(s, a) / N(s, a) 
        self.W = {}

        #: cache next states to save computation
        self.next_states = {}

        #: 辞書のkeyにするためにリストを文字列化する関数
        self.state_to_str = (
            lambda state, player: self.hash(state) + str(player)
            )
    
    def get_action_eval(self, state, action_mask, reverse):
        state = state.detach().numpy()
        player = self.env.current_player(state)
        policy = self.search(state, player, self.num_sims)
        action = random.choice(np.where(np.array(policy) == max(policy))[0])
        return action

    def search(self, root_state, current_player, num_simulations):
        
        #: dictのkeyにするために文字列化する
        s = self.state_to_str(root_state, current_player)
        
        #: 子盤面が無ければ展開
        if s not in self.P:
            _ = self._expand(root_state, current_player)

        valid_actions = self.get_valid_actions(root_state, current_player)

        #: root状態にだけは事前確立にディリクレノイズをのせて探索を促進する
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha]*len(valid_actions))
        for a, noise in zip(valid_actions, dirichlet_noise):
            self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        #: MCTS simulationの実行
        for _ in tqdm(range(num_simulations), leave=False):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_num)]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]
            
            #: PUCTスコアの算出
            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            #: スコアのもっとも高いactionを選択
            action = random.choice(np.where(scores == scores.max())[0])
            next_state = self.next_states[s][action]
            
            #: 選択した行動を評価（次は相手番なので評価値に-1をかける）
            v = -self._evaluate(next_state, 1-current_player)

            self.W[s][action] += v
            self.N[s][action] += 1
        
        #: mcts_policyは全試行回数に占める各アクションの試行回数の割合
        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def _expand(self, state, current_player):
        """ 子盤面の展開
        """
        s = self.state_to_str(state, current_player)
        
        nn_policy, nn_value = self.network(torch.FloatTensor(state))

        nn_policy, nn_value = nn_policy.detach().numpy().tolist()[0], nn_value.detach().numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * self.action_num
        self.W[s] = [0] * self.action_num

        valid_actions = self.get_valid_actions(state, current_player)

        #: パフォーマンスが向上のために次の状態を保存しておく
        self.next_states[s] = [
            self.next(state, action, current_player)[0]
            if (action in valid_actions) else None
            for action in range(self.action_num)]

        return nn_value

    def _evaluate(self, state, current_player):
        """盤面の評価
        """

        s = self.state_to_str(state, current_player)

        if self.is_done(state, current_player):
            #: この盤面でゲーム終了の場合は実報酬を返す
            #: win: 1, lose: -1, draw: 0
            reward_first, reward_second = self.result(state)
            reward = reward_first if current_player == 0 else reward_second
            return reward

        elif s not in self.P:
            #: この盤面でゲーム終了ではなく、かつこの盤面が未展開の場合は展開する
            #: この盤面のニューラルネット評価値を返す
            nn_value = self._expand(state, current_player)
            return nn_value

        else:
            #: この盤面でゲーム終了ではなく、かつこの盤面が展開済みの場合
            
            #: PUCTによってさらに子盤面を選択する
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_num)]
            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]

            valid_actions = self.get_valid_actions(state, current_player)

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            action = random.choice(np.where(scores == scores.max())[0])
            next_state = self.next_states[s][action]
            
            #: 選択した子盤面を評価（次は相手番なので評価値に-1をかける）
            v = -self._evaluate(next_state, 1-current_player)

            self.W[s][action] += v
            self.N[s][action] += 1

            return v

def selfplay(mcts: PVMCTS, num_sim: int, env, dirichlet_alpha=0.35, prob_act_th=4) -> List[Dict[str, Any]]:
    """
    TODO
    env.init
    env.hash
    env.next
    env.isdone
    env.result
    env.get_valid_action
    env.action_num
    env.current_player
    """
    data = []
    state = env.init()
    current_player = 0
    done = False
    i = 0

    while not done:
        mcts_policy = mcts.search(state, current_player, num_sim)

        if i <= prob_act_th:
            action = np.random.choice(
                range(env.action_num), p=mcts_policy)
        else:
            #: MCTSの試行回数がもっとも大きいアクションを選択
            #: np.argmaxを使うと同値maxの場合に選択が偏る
            action = random.choice(
                np.where(np.array(mcts_policy) == max(mcts_policy))[0])
        
        data.append({
            "state": state,
            "policy": mcts_policy,
            "reward": 0,
            "player": env.current_player(state)
        })

        next_state, done = env.next(state, action, current_player)

        state = next_state
        current_player = 1 - current_player

        i += 1

    reward_first, reward_second = env.result(state)

    for que in data:
        que["reward"] = reward_first if que["player"] == 0 else reward_second
    
    return data



class AlphaZero:
    def __init__(self, pv: PVNet, config: AlphaZeroConfig, env, eval_func):
        self.pv = pv
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.memory = ReplayMemory(buffer_size=config.buffer_size, batch_size=config.batch_size)
        self.optim = optim.Adam(self.pv.parameters())
        self.writer = SummaryWriter(log_dir=join("./tensorboard", config.log_name))
        self.log_n = config.log_n
        self.log_name = config.log_name
        self.num_sims = config.sim_n
        self.env = env
        self.alpha = config.dirichlet_alpha
        self.prob_act_th = config.prob_action_th
        self.epoch = config.epoch
        self.save_n = config.save_n
        self.eval_n = config.eval_n

        self.eval_func = eval_func

    def train(self):
        self.step = 0
        train_loop_n = self.epoch // 5
        for i in tqdm(range(train_loop_n), desc="[train loop]"):
            self.memory = ReplayMemory(self.buffer_size, self.batch_size)
            mcts = PVMCTS(self.pv, self.alpha, self.env, num_sims=self.num_sims)
            i = 0
            bar = tqdm(total=self.buffer_size, desc="[selfplay]", leave=False)
            while not self.memory.is_full:
                i += 1
                data = selfplay(mcts, self.num_sims, self.env, self.alpha, self.prob_act_th)
                self.memory.push_sequence(data)
                bar.set_postfix(step=i)
                bar.update(len(data))
            
            iter_n = 5 * self.buffer_size // self.batch_size
            bar = tqdm(range(iter_n), desc="[update]", smoothing=0.99, leave=False)
            for i in bar:
                result = self.optimize()
                bar.set_postfix(result)        


    def optimize(self):
        self.step += 1
        batch = self.memory.sample()
        state = torch.FloatTensor(batch["state"])
        mcts_policy = torch.FloatTensor(batch["policy"])
        reward = torch.FloatTensor(batch["reward"])

        net_policy, net_value = self.pv(state)

        td_error = reward - net_value
        value_loss = torch.sum(torch.square(td_error))

        policy_loss = -mcts_policy * torch.log(net_policy + 0.0001)
        policy_loss = torch.sum(
            policy_loss, axis=1, keepdims=True)

        loss = torch.mean(value_loss + policy_loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.step % self.log_n == 0:
            self.writer.add_scalar("loss/value_loss", torch.sum(value_loss).item(), self.step)
            self.writer.add_scalar("loss/policy_loss", torch.sum(policy_loss).item(), self.step)
            self.writer.add_scalar("loss/sum_loss", loss.item(), self.step)
        
        if self.step % self.eval_n == 0:
            result = self.eval_func(
                PVMCTS(self.pv, alpha=0.35, env=self.env, epsilon=0, num_sims=self.num_sims)
            )
            for key, val in result.items():
                self.writer.add_scalar(key, val, self.step)

        if self.step % self.save_n == 0:
            save_model(self.step, self.pv, self.log_name)
        
        return {
            "v_loss": torch.sum(value_loss).item(),
            "p_loss": torch.sum(policy_loss).item(),
            "loss": loss.item()
        }