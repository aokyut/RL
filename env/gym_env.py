import gym
from gym import wrappers
import requests
import json
from time import sleep
import os

# env上で動いているモデルに紐づいているクラス
class Model:
    def __init__(self, url, env, brain_config={}, is_render=True, debug=False):
        self.new_url = url + "/new"
        self.get_id(brain_config, self.new_url)
        self.run_url = url + "/run"
        self.env = env
        self.is_render = is_render
        self.debug = debug

        self.max_step = 0
    
    # モデルを作成する際にモデルのidを決定する
    def get_id(self, config, url):
        json_data = json.dumps(config)
        response = requests.post(url, json_data, headers={'Content-Type': 'application/json'})

        assert response.status_code == 200, "----- status code is " + str(response.status_code) + " -----"
        
        self.id = response.json()["id"]
        print(self.id)
    
    # 行動を獲得する。
    def act(self, state, reward, done):
        json_data = json.dumps({
            "id": self.id,
            "state": list(state),
            "reward": float(reward),
            "done": 1 if done is True else 0
        })
        response = requests.post(self.run_url, json_data, headers={'Content-Type': 'application/json'})

        assert response.status_code == 200, "----- status code is " + str(response.status_code) + " -----"

        return response.json()["action"]
    
    def finalize_env(self, state, reward, done):
        json_data = json.dumps({
            "id": self.id,
            "state": list(state),
            "reward": float(reward),
            "done": 1 if done is True else 0
        })
        response = requests.post(self.run_url, json_data, headers={'Content-Type': 'application/json'})

        assert response.status_code == 200, "----- status code is " + str(response.status_code) + " -----"

    
    def debug_out(self, text):
        with open("./debug/{}".format(str(self.id)), "a") as f:
            f.write(text)
    
    # 学習を開始する
    def run(self):
        # 環境モデルの初期化
        state = self.env.reset()
        if self.is_render is True:
            self.env.render()
        reward = 0
        step = 0
        done = False

        while True:
            sleep(0.1)
            step += 1
            # get action
            action = self.act(state, reward, done)

            # execute action
            state, reward, done, _ = self.env.step(action)
            
            reward = (reward / 10) + (state[0]/10)
            # reward = 0

            # if done is True:
            #     if step >= 195:
            #         reward += 1.0
            #     else:
            #         reward += -1.0
            # else:
            #     reward += 0.005


            if self.is_render is True:
                self.env.render()
            
            if done is True:
                self.finalize_env(state, reward, done)
                break
        
        self.env.close()
        
        if self.debug is True:
            self.debug_out(str(step) + "\n")

class Env:
    def __init__(self, video_path, env_name="CartPole-v0", video=False):
        self.video = False
        self.video_path = video_path
        self.env_name = env_name
        self.set_env()
    
    def set_env(self):
        self.env = gym.make(self.env_name)

        if self.video is True:
            self.env = wrappers.Monitor(self.env, self.video_path, video_callable=(lambda ep: True), force=True)

    def video_on(self):
        self.video = True
        self.set_env()
    
    def video_off(self):
        self.video = False
        self.set_env()

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
        



if __name__ == "__main__":
    video_root = "video"
    gym_env = Env(env_name="MountainCar-v0", video_path="test.mp4")
    config = {
        "input_size": 2,
        "hidden_size": 20,
        "output_size": 3
    }
    model = Model(
        url = "http://localhost:8080/",
        env = gym_env,
        brain_config=config,
        debug = True
    )
    epoch = 0
    while True:
        epoch += 1
        print(epoch)
        if epoch % 100 == 0:
            gym_env.video_path = os.path.join(video_root, str(epoch))
            gym_env.video_on()
            # model.is_render = True
        else:
            gym_env.video_off
            model.is_render = False
        model.run()

