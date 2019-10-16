import random
import numpy as np


class RandomAgent:
    def __init__(self, actions_num, env, turn):
        self._actions_num = actions_num
        self._env = env
        self._turn = turn

    def update(self, train=False):
        pass

    def reset(self):
        pass

    def choose_action(self, state, prev_reward, done, train=True):
        return self.choose_random_action(state)

    def choose_random_action(self, state):
        available_actions = self._env.available_actions(state, self._turn)

        if np.any(available_actions):
            action = random.randint(0, self._actions_num - 1)

            while not available_actions[action]:
                action = random.randint(0, self._actions_num - 1)

            return action
        else:
            return -1
