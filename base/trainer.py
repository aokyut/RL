from utils.error import *
from tqdm import tqdm

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
    
    with tqdm(range(learn_n)) as t:
        for i in t:
            observation = env.reset()
            reward = 0
            done = False
            # episode = []
            sum_reward = 0
            step = 0
            while True:
                step += 1
                action = model.get_action(observation)
                next_observation, next_reward, next_done, _ = env.step(action)
                memory.push(state=observation, action=action, reward=next_reward, next_state=next_observation, mask=float(not next_done))
                # episode.append(observation, next_observation, reward, done, action)
                sum_reward += next_reward
                if next_done:
                    # episode.append(next_observation, next_reward, nextt_done, action)
                    break
                observation = next_observation
            # memory.append(episode)
            model.optimize(writer, memory)
            writer.add_scalar("value/sum_reward", sum_reward, model.step)
            writer.add_scalar("value/step", step, model.step)
            t.set_postfix(reward=sum_reward, step=step)
