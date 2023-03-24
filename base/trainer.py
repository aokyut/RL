from utils.error import *
from tqdm import tqdm
from time import sleep
from flask import Flask, jsonify, request, Blueprint
import json
from collections import deque
from threading import Thread

#TODO
def gym_trainer(model, env, writer, memory, learn_n):
    """
    Parameters
    -----
    model: class(BaseModel)
    env: gym.env
    writer: summarywriter
    memory: class(BaseMemory)
    """
    if env.observation_space.shape != model.observation_space:
        raise ShapeError(f"env.observation_space.shape: {env.observation_space.shape}, model.observation_space:{model.observation_space}")
    if env.action_space.shape != model.action_space:
        raise ShapeError(f"env.action_space.shape: {env.action_space.shape}, model.action_space:{model.action_space}")
    sum_rewards = [0 for i in range(10)]
    step_list = [0 for i in range(10)]
    with tqdm(range(learn_n)) as t:
        for i in t:
            observation = env.reset()
            reward = 0
            done = False
            # episode = []
            sum_reward = 0
            step = 0
            while True:
                sleep(0.04)
                step += 1
                action = model.get_action(observation)
                next_observation, next_reward, next_done, _ = env.step(2 * action)
                memory.push(state=observation, action=action, reward=next_reward, next_state=next_observation, mask=float(not next_done))
                # episode.append(observation, next_observation, reward, done, action)
                sum_reward += next_reward
                if next_done:
                    # episode.append(next_observation, next_reward, nextt_done, action)
                    break
                observation = next_observation
            # memory.append(episode)
            model.optimize(writer, memory)
            sleep(0.5)
            sum_rewards.append(sum_reward)
            sum_rewards = sum_rewards[1:]
            step_list.append(step)
            step_list = step_list[1:]
            reward_mean = sum(sum_rewards) / 10
            step_mean = sum(step_list) / 10
            if model.step % model.config["log_interval"] == 0:
                writer.add_scalar("value/sum_reward", sum(sum_rewards) / 10, model.step)
                writer.add_scalar("value/step", sum(step_list) / 10, model.step)
            t.set_postfix(reward=reward_mean, step=step_mean)