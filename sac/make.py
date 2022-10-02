import json

a = {
    "input_dim": 5,
    "output_dim": 6,
    "hidden_size": 10,
    "hidden_layers": 3,
    "step": 0,
    "policy": "gaussian"
}

with open("default.json", "w") as f:
    json.dump(a, f, indent=4)