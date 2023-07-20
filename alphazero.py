from network import PVNet
from utills import ConfigParser, _save_model, huber_error, hard_update
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
import ray
from envs.base import play, ModelAgent

@dataclass
class AlphaZeroConfig:
    buffer_size: int = 2 << 15
    batch_size: int = 64
    log_name: str = "alphazero"
    log_n: int = 100
    save_n: int = 20000
    sim_n: int = 40
    prob_action_th: int = 3 #この手数に達するまでは探索回数に比例する確立で行動する
    dirichlet_alpha: float = 0.35
    episode: int = 100
    selfplay_n: int = 300
    use_ray: bool = False
    selfplay_para_n: int = 1 #selfplayの同時進行数
    epoch: int = 10 # 学習の際にバッファー何回分の学習を回すかの設定
    save_dir: str = "checkpoint"
    eval_best_n: int = 20 # bestモデルを評価する際に対戦する回数
    change_best_r: float = 0.55 # bestモデルを交代する際の閾値


class PVMCTS:
    def __init__(self, network, alpha, env, c_base=19652, c_init=1.00, epsilon=0.25, num_sims=50):
        self.network = network
        self.alpha = alpha
        self.c_base = c_base
        self.c_init = c_init
        self.eps = epsilon

        self.hash = env.hash
        self.next = env.get_next
        self.is_done = env.is_done
        self.result = env.result
        self.get_valid_actions = env.valid_actions
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
        # state = state.detach().numpy().astype(np.uint8)
        player = self.env.current_player(state)
        policy = self.search(state, player, self.num_sims, True)
        action = random.choice(np.where(np.array(policy) == max(policy))[0])
        return action
    
    def analyze_and_action(self, state, action_mask, reverse):
        # current_state = state.detach().numpy().astype(np.uint8)
        current_state = state.astype(np.uint8)
        player = self.env.current_player(state)
        policy = self.search(current_state, player, self.num_sims, False)
        action = random.choice(np.where(np.array(policy) == max(policy))[0])
        # action = self.get_action_eval(torch.tensor(state), action_mask, reverse)
        valid_actions = self.env.get_valid_action(state, player)
        analysys = []
        s = self.state_to_str(state, player)
        for valid_action in valid_actions:
            next_state, _ = self.env.next(state, valid_action, player)
            n = self.N[s][valid_action]
            w = self.W[s][valid_action]
            p = self.P[s][valid_action]
            analysys.append({"action":valid_action, "N":n, "P":p, "W":w})
        
        analysys = sorted(analysys, key=lambda x: -x["N"])
        n_sum = sum([que["N"] for que in analysys])
        for que in analysys:
            act = que["action"]
            n = que["N"]
            p = que["P"]
            w = que["W"]
            print("{:5d}: {:7.1f}% ({:5d})(N), ".format(act, 100 * n/n_sum, n), end="")
            q = (w / n) if n != 0 else 0
            if w < 0:
                print('\033[31m' + "{:7.4f}".format(q) + '\033[0m' + "(Q), ", end="")
            else:
                print('\033[32m' + "{:7.4f}".format(q) + '\033[0m' + "(Q), ", end="")
            print("{:7.4f}(P_net)".format(p))


    def search(self, root_state, current_player, num_simulations, disable_tqdm=True):
        
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
        # sim_n = max(num_simulations - sum(self.N[s]), 0)

        #: MCTS simulationの実行
        for _ in tqdm(range(num_simulations), leave=False, disable=disable_tqdm):
            cs = self.c_init + math.log((1 + sum(self.N[s]) + self.c_base) / self.c_base)
            U = [cs * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_num)]
            # U = [cs * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                #  for a in range(self.action_num)]
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
            self.next(state, action, current_player)
            if (action in valid_actions) else None
            for action in range(self.action_num)]

        return nn_value

    def _evaluate(self, state, current_player):
        """盤面の評価
        """

        s = self.state_to_str(state, current_player)

        if self.is_done(state, 1 - current_player):
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
            cs = self.c_init + math.log((1 + sum(self.N[s]) + self.c_base) / self.c_base)
            U = [cs * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(self.action_num)]
            # U = [cs * 1 * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
            #      for a in range(self.action_num)]
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

# @ray.remote(num_cpus=1)
def selfplay(network:PVNet, num_sim: int, env, dirichlet_alpha=0.35, prob_act_th=4) -> List[Dict[str, Any]]:
    """
    TODO
    env.init() -> ndarray
    env.hash(state) -> str
    env.next(state, action, player) -> ndarray
    env.isdone(state, player) -> bool
    env.result(state) -> [int, int]
    env.get_valid_action(state, player) -> List[int]
    env.action_num -> int
    env.current_player(state) -> int
    """
    data = []
    state = env.init()
    mcts = PVMCTS(network, alpha=dirichlet_alpha, env=env)
    current_player = 0
    done = False
    i = 0

    while not done:
        mcts_policy = mcts.search(state, current_player, num_sim)

        if i < prob_act_th:
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

        next_state = env.get_next(state, action, current_player)
        done = env.is_done(next_state, current_player)

        state = next_state
        current_player = 1 - current_player

        i += 1

    reward_first, reward_second = env.result(state)

    for que in data:
        que["reward"] = reward_first if que["player"] == 0 else reward_second
    
    return data

# rayを使用する際
selfplay_para = ray.remote(num_cpu=1)(selfplay)


def play(agent1, agent2, env):
    state = env.init()
    agents = [agent1, agent2]
    player = 0
    while True:
        agent = agents[player]
        action = agent.get_action_eval(state, None, None)
        next_state = env.get_next(state, action, player)
        state = next_state
        if env.is_done(next_state, player):
            if env.is_win(next_state, player):
                if player == 0:
                    return 1
                else:
                    return -1
            else:
                return 0
        player = 1 - player

def eval_model_agents(model1, model2, env, n, alpha, num_sims):
    win_1 = 0
    draw = 0
    win_2 = 0
    for i in tqdm(range(n // 2), leave=False, desc="[change model check]"):
        agent1 = PVMCTS(model1, alpha, env, num_sims=num_sims)
        agent2 = PVMCTS(model2, alpha, env, num_sims=num_sims)
        result = play(agent1, agent2, env)
        if result == 1:
            win_1 += 1
        elif result == -1:
            win_2 += 1
        else:
            draw += 1
        agent1 = PVMCTS(model1, alpha, env, num_sims=num_sims)
        agent2 = PVMCTS(model2, alpha, env, num_sims=num_sims)
        result = play(agent2, agent1, env)
        if result == 1:
            win_2 += 1
        elif result == -1:
            win_1 += 1
        else:
            draw += 1
    return (win_1 + draw / 2) / n, (win_2 + draw / 2) / n
        

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
        self.episode = config.episode
        self.save_n = config.save_n
        self.selfplay_n = config.selfplay_n
        self.selfplay_para_n = config.selfplay_para_n

        self.eval_func = eval_func
        self.use_ray = config.use_ray
        self.epoch = config.epoch
        self.save_dir = config.save_dir

        self.eval_best_n = config.eval_best_n
        self.change_best_r = config.change_best_r
        self.best_model = self.pv.clone()
        hard_update(self.best_model, self.pv)
        self.best_gen = 0

    def train(self):
        # rayを使用する時
        if self.use_ray:
            ray.init(num_cpus=2)
            pv = ray.put(self.pv)
            work_in_progresses = [
                selfplay_para.remote(
                    pv,
                    self.num_sims, 
                    env=self.env, 
                    dirichlet_alpha=self.alpha, 
                    prob_act_th=self.prob_act_th)
                for _ in range(self.selfplay_para_n)]


        self.step = 0
        train_loop_n = self.episode
        for i in tqdm(range(train_loop_n), desc="[train loop]", smoothing=0.999):
            self.pv.eval()

            result = self.eval_func(
               PVMCTS(self.pv, alpha=0.35, env=self.env, epsilon=0, num_sims=self.num_sims)
            )
            print(result)
            for key, val in result.items():
                self.writer.add_scalar(key, val, i * self.selfplay_n)

            # self.memory = ReplayMemory(self.buffer_size, self.batch_size)
            if not self.use_ray:
                mcts = PVMCTS(self.pv, self.alpha, self.env, num_sims=self.num_sims)

            bar = tqdm(total=self.selfplay_n, desc="[selfplay]", leave=False, smoothing=0.99)
            for _ in  range(self.selfplay_n):
                if self.use_ray:
                    finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
                    data = ray.get(finished[0])
                    work_in_progresses.extend([
                        selfplay_para.remote(
                            pv,
                            self.num_sims, 
                            env=self.env, 
                            dirichlet_alpha=self.alpha, 
                            prob_act_th=self.prob_act_th)])
                else:
                    data = selfplay(self.pv, self.num_sims, self.env, self.alpha, self.prob_act_th)
                self.memory.push_sequence(data)
                bar.update(1)
            bar.close()
            

            iter_n = self.epoch * len(self.memory.buffer) // self.batch_size
            bar = tqdm(range(iter_n), desc="[update]", smoothing=0.99, leave=False)
            self.pv.train()
            for i in bar:
                result = self.optimize()
                bar.set_postfix(result)

            self.pv.eval()

            # win_rate_best, win_rate_new = eval_model_agents(self.best_model, self.pv, self.env, self.eval_best_n, self.alpha, self.num_sims)
            # self.writer.add_scalar("eval/win_rate_new", win_rate_new, i * self.selfplay_n)
            # if win_rate_new >= self.change_best_r:
            #     print(f"change best model:{self.best_gen} -> {i}")
            #     self.best_gen = i
            #     hard_update(self.best_model, self.pv)
            # else:
            #     hard_update(self.pv, self.best_model)
            # self.writer.add_scalar("eval/best_model_gen", self.best_gen, i * self.selfplay_n)
        
            _save_model(self.save_dir, self.log_name, "latest", self.pv)



    def optimize(self):
        self.step += 1
        batch = self.memory.sample()
        state = torch.FloatTensor(batch["state"])
        mcts_policy = torch.FloatTensor(batch["policy"])
        reward = torch.FloatTensor(batch["reward"]).reshape(-1, 1)

        net_policy, net_value = self.pv(state)

        td_error = reward - net_value
        value_loss = torch.square(td_error)
        # value_loss = huber_error(torch.square(td_error))

        policy_error = -mcts_policy * torch.log(net_policy + 0.0001)
        policy_loss = torch.sum(
            policy_error, axis=1, keepdims=True)

        loss = torch.mean(value_loss + policy_loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.step % self.log_n == 0:
            self.writer.add_scalar("loss/value_loss", torch.mean(value_loss).item(), self.step)
            self.writer.add_scalar("loss/policy_loss", torch.mean(policy_loss).item(), self.step)
            self.writer.add_scalar("loss/sum_loss", loss.item(), self.step)

        if self.step % self.save_n == 0:
            _save_model(self.save_dir, self.log_name, self.step, self.pv)
        
        return {
            "v_loss": "{:7.3f}".format(torch.sum(value_loss).item()),
            "p_loss": "{:7.3f}".format(torch.sum(policy_loss).item()),
            "loss": "{:7.3f}".format(loss.item())
        }
