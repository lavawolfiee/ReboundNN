import copy
import numpy as np
import random
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import math
from Agent import *


class NNAgent(Agent):
    def __init__(self, states_num, actions_num, env, turn, memory_size=50000, batch_size=1000, train_epochs=1,
                 learning_rate=0.1, discount=0.95, exploration_rate=1, iterations=100000, actions_in_iter=32):
        self._memory = []
        self._memory_size = memory_size
        self._states_num = states_num
        self._actions_num = actions_num
        self._batch_size = batch_size
        self._train_epochs = train_epochs
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration_rate = exploration_rate
        self._iterations = iterations
        self._exploration_delta = 1/self._iterations/actions_in_iter
        self._model = self._define_model()
        self._prev_state = np.zeros(self._states_num)
        self._prev_action = -1
        self._prev_reward = 0
        self._env = env
        self._turn = turn

    def _define_model(self, weights=None):
        model = Sequential()
        model.add(Dense(360, activation='linear', input_dim=self._states_num))
        model.add(Dropout(0.15))
        model.add(Dense(360, activation='linear'))
        model.add(Dropout(0.15))
        model.add(Dense(360, activation='linear'))
        model.add(Dropout(0.15))
        model.add(Dense(360, activation='linear'))
        model.add(Dropout(0.15))
        model.add(Dense(self._actions_num, activation='linear'))
        opt = Adam(self._learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def predict_one(self, state):
        return self._model.predict(state.reshape(1, self._states_num))[0]

    def predict_batch(self, states):
        return self._model.predict(states)

    def train_batch(self, x_train, y_train):
        self._model.fit(x_train, y_train, epochs=self._train_epochs, verbose=0)

    def add_sample(self, sample):
        self._memory.append(sample)
        if len(self._memory) > self._memory_size:
            self._memory.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._memory):
            return random.sample(self._memory, len(self._memory))
        else:
            return random.sample(self._memory, no_samples)

    def update(self, train=True):
        self._exploration_rate = max(0.0, self._exploration_rate - self._exploration_delta)
        if train:
            self.replay()

    def reset(self):
        self._prev_state = np.zeros(self._states_num)
        self._prev_action = -1
        self._prev_reward = 0

    def choose_action(self, state, prev_reward, done, train=True):
        if self._prev_action != -1 and train:
            self.add_sample((self._prev_state, self._prev_action, prev_reward, (None if done else state),
                             copy.deepcopy(self._env.pieces)))

        if not done:
            if random.random() < self._exploration_rate and train:
                action = self.choose_random_action(state)
            else:
                action, value = self.choose_optimal_action(state, self._env.pieces)

            self._prev_action = action
            self._prev_state = state

            return action

        return -1

    def choose_random_action(self, state):
        available_actions = self._env.available_actions(state, self._turn)

        if np.any(available_actions):
            action = random.randint(0, self._actions_num - 1)

            while not available_actions[action]:
                action = random.randint(0, self._actions_num - 1)

            return action
        else:
            print('no moves!')
            return -1

    def choose_optimal_action(self, state, pieces):
        available_actions = self._env.available_actions(state, turn=self._turn, pieces=pieces)
        v = 0
        flag = True
        ids = []
        pred = self.predict_one(state)

        for action in range(0, self._actions_num):
            if math.isnan(pred[action]):
                raise Exception("Wrong pred: " + str(pred) + " in state: " + str(state) + ". ")
            if available_actions[action]:
                t = pred[action]

                if flag or t > v:
                    flag = False
                    v = t
                    ids = [action]
                elif t == v:
                    ids.append(action)

        if len(ids) == 0:
            return -1, 0
        else:
            return random.choice(ids), v

    def replay(self):
        if len(self._memory) > 0:
            batch = self.sample(self._batch_size)
            old_states = np.array([val[0] for val in batch])
            old_q = self.predict_batch(old_states)

            x = np.zeros((len(batch), self._states_num))
            y = np.zeros((len(batch), self._actions_num))

            for i, b in enumerate(batch):
                prev_state, action, reward, state, pieces = b

                q = old_q[i]

                if state is None:
                    q[action] = reward
                else:
                    n_action, n_value = self.choose_optimal_action(state, pieces)
                    if n_action != -1:
                        q[action] = reward + self._discount * n_value
                    else:
                        q[action] = reward

                x[i] = prev_state
                y[i] = q

            self.train_batch(x, y)
