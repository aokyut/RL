import gym
from .networks import GaussianPolicy
from .utils import Transition, SacConfig, parse_from_dataclass
from typing import List
import numpy as np
from .agent import SACAgent
import torch
from tqdm import tqdm
import sys


# env = gym.make("BipedalWalker-v3")
env = gym.make("Pendulum-v1")
episode_end = 200

class ModelAgent:
    def __init__(self, model:GaussianPolicy):
        self.model = model
    
    def get_action(self, state):
        action, _ = self.model(torch.FloatTensor(state))
        action = action.to("cpu").detach().numpy().copy().reshape((-1))
        return action
    

def explore_func(model:GaussianPolicy, use_model=True) -> List[Transition]:
    seq = []
    agent = ModelAgent(model)
    state, _ = env.reset()
    bar = tqdm(leave=False, file=sys.stdout)
    step = 0
    while True:
        bar.update(1)
        step += 1
        if use_model:
            action = agent.get_action(state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

        if step >= episode_end:
            done = True
        seq.append(
            Transition(
                states=state,
                actions=action,
                rewards=np.array([reward]),
                next_states=next_state,
                dones=np.array([1 if done else 0])
            )
        )
        bar.set_postfix(reward=reward, done=done)
        state = next_state
        if done:
            break
    bar.close()
    return seq

def eval_func(model:GaussianPolicy):
    acum_reward = 0
    eval_n = 10
    agent = ModelAgent(model)
    for i in tqdm(range(eval_n), leave=False, file=sys.stdout):
        state, _ = env.reset()
        step = 0
        while True:
            step += 1
            # action = env.action_space.sample()
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            acum_reward += reward
            if step >= episode_end:
                done = True
            if done:
                break
            state = next_state
    
    return {"eval/acum_reward": acum_reward / eval_n}


config = SacConfig(
    state_shape=[3],
    action_size=1,
    buffer_size=10_000,
    batch_size=32,
    target_hard_update_n=1000_000,
    tar_update_tau=0.005,
    tar_update_n=1,
    hidden_size=64,
    n_block=3,
    update_per_episode=20,
    start_alpha=0.2,
    episode_n=1_000,
    log_name="sac-hurber",
    log_dir="./exp_tensorboard",
    lr_alpha=0.0001
    # save_n=5,
    # log_n=10
)

if __name__ == "__main__":
    config = parse_from_dataclass(SacConfig, config)
    print(config)
    learn_agent = SACAgent(config, eval_func, explore_func)
    learn_agent.policy.eval()
    # explore_func(learn_agent.policy)
    # print(eval_func(learn_agent.policy))
    
    learn_agent.train()