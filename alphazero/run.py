from network import AlphaZeroNetwork
from mtcs import AlphaZeroAgent, PlayerAgent
from renju import play, Game
import torch
import argparse
import config
from os import path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_model")
    parser.add_argument("--load_dir", default=config.save_dir)
    parser.add_argument("--mtcn_n", type=int, default=config.azero_puct_n)

    args = parser.parse_args()

    p = PlayerAgent()
    network = AlphaZeroNetwork()

    network.load_state_dict(torch.load(path.join(args.load_dir, args.load_model)))

    a = AlphaZeroAgent(network, args.mtcn_n)
    Game(a)
    # play(a, p)
