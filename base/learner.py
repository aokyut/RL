from utils.error import *

#TODO
def gym_trainer(model, env, optimizer, writer, memory, episode_n, learn_n):
    """
    Parameters
    -----
    model: class(BaseModel)
    env: gym.env
    optimizer: pytorch.optim
    writer: summarywriter
    memory: class(BaseMemory)
    """
    if env.observation_space.shape != model.observation_space:
        raise ShapeError(f"env.observation_space.shape: {env.observation_space.shape}, model.observation_space:{model.observation_space}")
    if env.action_space.n != model.action_space:
        raise ShapeError(f"env.action_space.n: {env.action_space.n}, model.action_space:{model.action_space}")
    
    for i in range(learn_n):
        observation = env.reset()
        reward = 0
        done = False
        episode = []
        while True:
            action = model.get_action(observation)
            next_observation, next_reward, next_done, _ = env.step(action)
            episode.append(observation, reward, done, action)
            if done:
                episode.append(next_observation, next_reward, nextt_done, action)
                break
        memory.append(episode)
        model.optimize(writer, memory, optimizer)
