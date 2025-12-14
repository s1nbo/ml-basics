import numpy as np
import pickle
from collections import defaultdict


class RandomAgent:
    def __init__(self, name):
        self.name = name

    def get_action(self, observation):
        valid_actions = np.where(observation == 0)[0]
        return np.random.choice(valid_actions)


class RlAgent: