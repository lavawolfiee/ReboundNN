from Environment import *
import numpy as np


class ReboundEnv(Environment):
    def __init__(self, width, height, pieces_count=2, reset=False):
        if not reset:
            self.KILL_REWARD = 1
            self.WIN_REWARD = 10
            self.LOSE_REWARD = -5
            self.DRAW_REWARD = 2

        self._width = width
        self._height = height
        self._field = np.zeros((self._height, self._width))
        if not reset:
            self.pieces_num = 2 # number of different piece's colors
            self.pieces_count = pieces_count # count of pieces of one color
            self.states_num = self._width * + self._height
            self.actions_num = self.states_num * self.pieces_num

        self.pieces = np.full((self.pieces_num * 2), self.pieces_count)  # 1 2 -1 -2
        self._turn = 1
        self._done = False
        self.scores = [0, 0]
        self.winner = 0

    def get_state(self):
        return np.multiply(self._field.reshape((-1)), self._turn)

    def step(self, action):
        state = self.get_state()
        piece, cell, x, y = self.break_action(action)

        if not self._done:
            if state[cell] != 0:
                raise WrongActionException("", state, action)

            piece_pos = abs(piece) - 1 + (self.pieces_num if self._turn == -1 else 0)
            if self.pieces[piece_pos] > 0:
                self.pieces[piece_pos] -= 1
            else:
                raise WrongActionException("You don't have pieces " + str(piece) + ", pieces: " + str(self.pieces) +
                                           ". ", state, action)

            self._done |= self.check_pieces()

            reward = 0

            # 1 beat -1, -1 beat 2, 2 -> -2, ..., n -> -n, -n -> n + 1, ..., -k -> 1 (n is natural, k = pieces_num)
            piece_to_beat = (-piece if piece > 0 else (1 if piece == -self.pieces_num else -piece + 1))
            beat_dirs = [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
            self._field[y][x] = piece

            for direction in beat_dirs:
                cx = x + direction[0]
                cy = y + direction[1]

                if not self._in_bound(cx, cy):
                    continue

                if self._field[cy][cx] == piece_to_beat:
                    nx = cx + direction[0]
                    ny = cy + direction[1]

                    if not self._in_bound(nx, ny):
                        reward += self.KILL_REWARD
                        self.scores[int(abs((self._turn - 1) / 2))] += self.KILL_REWARD
                        self._field[cy][cx] = 0
                    elif self._field[ny][nx] == 0:
                        self._field[ny][nx] = piece_to_beat
                        self._field[cy][cx] = 0

            self._turn *= -1
            return self.get_state(), reward, self._done
        '''else:
            reward = self.check_winner()
            self._turn *= -1
            return self.get_state(), reward, self._done'''

    def reset(self):
        self.__init__(self._width, self._height, reset=True)
        pass

    def check_pieces(self):
        return not self.pieces.any()

    def check_action(self, x, y, piece, state):
        cell = self._width * y + x
        piece_pos = abs(piece) - 1 + (self.pieces_num if self._turn == -1 else 0)

        if state[cell] != 0 or self.pieces[piece_pos] <= 0:
            return False
        else:
            return True

    def to_action(self, x, y, piece):
        return y * self._width + x + (self._width * self._height if piece == 2 else 0)

    def break_action(self, action):
        piece = (action // self.states_num + 1) * self._turn
        cell = action % self.states_num
        x = cell % self._width
        y = cell // self._width

        return piece, cell, x, y

    def check_winner(self):
        if self.scores[0] > self.scores[1]:
            self.winner = 1
        elif self.scores[0] < self.scores[1]:
            self.winner = -1
        else:
            self.winner = 0

    def get_end_reward(self, turn):
        self.check_winner()
        return self.DRAW_REWARD if self.winner == 0 else self.WIN_REWARD if turn == self.winner else self.LOSE_REWARD

    def _in_bound(self, x, y):
        return (x >= 0) and (y >= 0) and (x < self._width) and (y < self._height)

    def available_actions(self, state, turn=None, pieces=None):
        if turn is None:
            turn = self._turn
        if pieces is None:
            pieces = self.pieces

        pieces_shift = self.pieces_num if turn == -1 else 0
        f = np.asarray(state == 0).reshape((-1)) & (pieces[pieces_shift] > 0)
        s = np.asarray(state == 0).reshape((-1)) & (pieces[pieces_shift + 1] > 0)
        return np.append(f, s)

    def __str__(self):
        b = "+" + "-" * (self._width * 2 - 1) + "+\n"
        s = b

        for y in range(self._height):
            s += "|"
            for x in range(self._width):
                s += str(int(self._field[y][x]))
                if x < self._width - 1:
                    s += " "
                else:
                    s += "|\n"

        s += b
        return s

    def render(self):
        pass


class WrongActionException(Exception):
    def __init__(self, msg, state, action):
        self.action = action
        self.state = state
        self.msg = msg

    def __str__(self):
        return self.msg + "Wrong action: " + str(self.action) + ", in state: " + str(self.state)
