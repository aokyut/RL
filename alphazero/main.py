from renju import Game
from mtcs import RandomAgent, UCTAgent

r = RandomAgent()
u = UCTAgent(30000)
Game(u, True)