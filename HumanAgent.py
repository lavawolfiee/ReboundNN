from Agent import *


class HumanAgent(Agent):
    def __init__(self, actions_num, env):
        self._actions_num = actions_num
        self._env = env

    def update(self, train=False):
        pass

    def reset(self):
        pass

    def choose_action(self, state, prev_reward, done, train=True):
        print("Human turn\n")
        print("Scores: " + str(self._env.scores))
        print("Pieces: " + str(self._env.pieces))
        print(self._env)

        x, y, piece = self._prompt_action()

        while not self._env.check_action(x, y, piece, state):
            print("Wrong action!")
            x, y, piece = self._prompt_action()

        return self._env.to_action(x, y, piece)

    def _prompt_action(self):
        return map(int, input("\nChoose your move (x, y, cell): ").split())
