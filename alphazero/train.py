import ray
from tqdm import tqdm
from network import AlphaZeroNetwork
from buffer import ReplayBuffer, Sample
from mtcs import PVmtcs, AlphaZeroAgent, RandomAgent, UCTAgent
from renju import get_init, get_next_state, get_action_mask, ACTION_SPACE, eval_play
import config

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import random
from os import path, makedirs
import argparse
import warnings

warnings.filterwarnings("ignore")

NUM_CPU = multiprocessing.cpu_count()
BATCH_SIZE = config.batch_size
LR = config.lr
BUFFER_SIZE = config.buffer_size


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_sims, dirichlet_alpha=0.35):
    record = []

    network = AlphaZeroNetwork()
    network.load_state_dict(weights)

    mtcs = PVmtcs(network, dirichlet_alpha)
    state = get_init()

    done = False
    i = 0
    current_player = 0
    reward = [0, 0]
    while not done:
        i += 1
        mtcs_policy = mtcs.search(state, num_sims)

        if i < 4:
            action = np.random.choice(
                range(ACTION_SPACE), p=mtcs_policy
            )
        else:
            action = random.choice(
                np.where(np.array(mtcs_policy) == max(mtcs_policy))[0]
            )

        record.append(Sample(state, mtcs_policy, current_player, None, get_action_mask(state)))
        next_state, done, iswin = get_next_state(state, action)

        if iswin:
            reward[current_player] = 1
            reward[1 - current_player] = -1

        state = next_state
        current_player = 1 - current_player

    for sample in record:
        sample.reward = reward[0] if current_player == 0 else reward[1]
    return record, reward[0], i


def main(args):
    n_parallel_selfplay = args.parallel_n
    ray.init(num_cpus=NUM_CPU)
    num_mtcs_sims = args.selfplay_sim_puct_num

    writer = SummaryWriter(path.join(args.log_dir, args.log_name))

    network = AlphaZeroNetwork()
    best_model = AlphaZeroNetwork()
    best_model.load_state_dict(network.state_dict())
    best_model.eval()

    current_weights = ray.put(network.state_dict())
    # current_weights = network.state_dict()
    optimizer = optim.SGD(network.parameters(), lr=LR)

    work_in_progress = [
        selfplay.remote(current_weights, num_mtcs_sims)
        for _ in range(n_parallel_selfplay)
    ]

    n = 0
    learn_step = 0
    while n < args.selfplay_total:
        player_0_win = 0
        player_1_win = 0
        play_step = 0
        replay = ReplayBuffer(buffer_size=args.buffer_size)
        for _ in tqdm(range(args.selfplay_num), smoothing=0.1, desc=f"[selfplay:{n}~{args.selfplay_num + n}]"):
            # selfplay(current_weights, num_mtcs_sims)
            finished, work_in_progress = ray.wait(work_in_progress, num_returns=1)

            record, win, step = ray.get(finished[0])
            replay.add_record(record)
            if win == 1:
                player_0_win += 1
            elif win == -1:
                player_1_win += 1

            play_step += step

            work_in_progress.extend([
                selfplay.remote(current_weights, num_mtcs_sims)
            ])

            n += 1
        writer.add_scalar("eval/player_0_win", player_0_win / args.selfplay_num, n)
        writer.add_scalar("eval/player_1_win", player_1_win / args.selfplay_num, n)
        writer.add_scalar("eval/play_step", play_step / args.selfplay_num, n)

        network.train()

        dataloader = DataLoader(replay,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)
        for i in tqdm(range(5), leave=False):
            for states, masks, mtcs_policy, reward in tqdm(dataloader, desc=f"[update: data{len(replay)}]", leave=False):
                learn_step += 1

                p, v = network(states, masks)

                value_loss = F.mse_loss(v, reward)

                policy_loss = - mtcs_policy * torch.log(p + 0.0001)
                policy_loss = torch.mean(policy_loss)

                loss = torch.mean(policy_loss + value_loss)

                optimizer.zero_grad()
                loss.backward()

                if learn_step % args.log_step:
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), learn_step)
                    writer.add_scalar("loss/value_loss", value_loss.item(), learn_step)
                    writer.add_scalar("loss/loss", loss.item(), learn_step)

        network.eval()

        # eval_step
        agent_r = RandomAgent()
        agent_u = UCTAgent(args.eval_uct_n)
        agent_a = AlphaZeroAgent(best_model, args.azero_puct_n)
        agent_tar = AlphaZeroAgent(network, args.azero_puct_n)

        r_rate = eval_play(agent_tar, agent_r, args.eval_play_n)
        u_rate = eval_play(agent_tar, agent_u, args.eval_play_n)
        a_rate = eval_play(agent_tar, agent_a, args.eval_play_n)

        writer.add_scalar("eval/vs_random", r_rate[0], n)
        writer.add_scalar(f"eval/vs_UCT{args.eval_play_n}", u_rate[0], n)
        writer.add_scalar("eval/vs_best", a_rate[0], n)

        if a_rate[0] > 0.5:
            print(f"model update {n}")
            best_model.load_state_dict(network.state_dict())
            if not path.exists(args.save_dir):
                makedirs(args.save_dir)
            save_name = path.join(args.save_dir, args.log_name + str(n))
            torch.save(network.state_dict(), save_name)

        current_weights = ray.put(network.state_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=config.log_dir)
    parser.add_argument("--save_dir", default=config.save_dir)
    parser.add_argument("--log_name", default=config.log_name)
    parser.add_argument("--batch_size", default=config.batch_size, type=int)
    parser.add_argument("--buffer_size", default=config.buffer_size, type=int)
    parser.add_argument("--log_step", type=int, default=config.log_step)
    parser.add_argument("--eval_play_n", type=int, default=config.eval_play_n)
    parser.add_argument("--eval_uct_n", type=int, default=config.eval_uct_n)
    parser.add_argument("--selfplay_num", type=int, default=config.selfplay_num)
    parser.add_argument("--selfplay_sim_puct_num", type=int, default=config.selfplay_sim_puct_num)
    parser.add_argument("--azero_puct_n", type=int, default=config.azero_puct_n)
    parser.add_argument("--parallel_n", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--selfplay_total", type=int, default=config.selfplay_total)

    args = parser.parse_args()

    main(args)
