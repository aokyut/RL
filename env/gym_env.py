import gym
import requests
import json

# env上で動いているモデルに紐づいているクラス
class Model:
    def __init__(self, env="MountainCar-v0", brain_config={}, url):
        self.new_url = url + "/new"
        self.get_id(brain_config, self.new_url)
        self.run_url = url + "/run"

    
    
    def get_id(self, config, url):
        json_data = json.dumps(config)
        response = requests.post(url, json_data, headers={'Content-Type': 'application/json'})

        assert response.status_code == 200, "status code is " + str(response.status_code)
        
        self.id = response.json()["id"]
    
    def act(self, state, reward):
        json_data = json.dumps({
            "id": self.id,
            "state": state,
            "reward": reward 
        })
        response = requests.post(self.run_url, json_data, headers={'Content-Type': 'application/json'})

        assert response.status_code == 200, "status code is " + str(response.status_code)

        return response["action"]
