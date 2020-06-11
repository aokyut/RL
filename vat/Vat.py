from flask import Flask, request, jsonify
import json
from Config import Config
from Brain import Brain
from Agent import Agent

app = Flask(__name__)
agents = []
brains = {}

@app.route("/", methods=["GET"])
def hello_check():
    return "ok"

@app.route("/new", methods=["POST"])
def new_agent():
    config = Config()
    req_data = json.loads(request.data)
    config.set(req_data)

    brain = get_brain(config)
    agent = Agent(brain, train=config.is_learn)
    res_data = {"id": len(agents)}
    agents.append(agent)
    return jsonify(res_data)

@app.route("/run", methods=["POST"])
def get_action():
    req_data = json.loads(request.data)
    agent_id = req_data["id"]
    state = req_data["state"]
    reward = req_data["reward"]
    res_data = {"action": agents[agent_id].step(state, reward)}
    jsonify(res_data)

# config通りのbrainが存在したら返す。存在しない場合は作成して返す
def get_brain(config):
    hash = str(config)

    if hash in brains:
        return brains[hash]

    brain = Brain(config)
    brains[hash] = brain
    return brain

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
