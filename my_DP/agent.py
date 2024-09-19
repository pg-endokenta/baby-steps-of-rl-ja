import random

from environment import Environment

class Agent():
    
    def __init__(self, env: Environment):
        self.actions = env.actions

    def policy(self, state):
        """ actionを返す """
        return random.choice(self.actions)
