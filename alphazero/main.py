from renju import Game, play, eval_play
from mtcs import RandomAgent, UCTAgent, AlphaZeroAgent
from network import AlphaZeroNetwork

network = AlphaZeroNetwork()
agent = AlphaZeroAgent(network, 300)
r = RandomAgent()

a = eval_play(r, agent, 20)
print(a[0], a[1])
