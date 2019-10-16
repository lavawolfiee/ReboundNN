import json
from ReboundEnv import *
from NNAgent import *
from RandomAgent import *
from HumanAgent import *
import argparse


def play(env, agent1, agent2, iterations=None, output=False, train=True, render=False, performance_test=True):
    i = 0
    agents = [agent1, agent2]  # 1 -1
    wins = [0, 0, 0]  # -1 0 1

    while iterations is None or (i < iterations):
        env.reset()
        state = env.get_state()
        done = False
        flag = False
        rewards = [0, 0]
        j = 0
        done_counter = 2
        if performance_test:
            pass
            # print('Iter ' + str(i))

        while done_counter > 0:
            if done:
                if performance_test:
                    done_counter -= 1
                else:
                    break

            if render:
                env.render()

            action = agents[j].choose_action(state, rewards[j], done, train=train)
            if not done:
                state, rewards[j], done = env.step(action)

            if train and done and not flag:
                flag = True
                rewards[0] += env.get_end_reward(1)
                rewards[1] += env.get_end_reward(-1)

            agents[j].update(train=train)

            j = (j + 1) % 2

        env.check_winner()
        wins[env.winner + 1] += 1

        if output:
            print("End of game. Winner: " + str(env.winner) + "\n")

        i += 1

        if i % 10 == 0 and performance_test:
            _wins1, draws1 = test_performance(env, agent1, 1)
            # print("" + str(_wins) + "% (draws: " + str(draws) + "%)")
            _wins2, draws2 = test_performance(env, agent2, -1)
            # print("Performance of agent2: " + str(_wins) + "% (draws: " + str(draws) + "%)")
            print(json.dumps({'step': i, 'performance': _wins1, 'performance2': _wins2}))
    return wins


def test_performance(env, agent, turn):
    random_agent = RandomAgent(width * height * pieces_num, env, turn * -1)
    agents = [agent, random_agent]
    if turn == -1:
        agents = agents[::-1]

    wins = play(env, agents[0], agents[1], iterations=100, train=False, performance_test=False)
    games = sum(wins)
    return wins[turn + 1] * 100 / games, wins[1] * 100 / games


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=500, help='Iterations count')
parser.add_argument('--learning_rate', type=float, default=0.1, help='How quickly the algorithm tries to learn')
FLAGS, unparsed = parser.parse_known_args()

width = 3
height = 3
pieces_count = 2
pieces_num = 2
iterations = FLAGS.iterations

env = ReboundEnv(width, height, pieces_count=pieces_count)
agent1 = NNAgent(width * height, width * height * pieces_num, env, 1,
                 iterations=iterations, actions_in_iter=pieces_count * 2, learning_rate=FLAGS.learning_rate)
agent2 = NNAgent(width * height, width * height * pieces_num, env, -1,
                 iterations=iterations, actions_in_iter=pieces_count * 2, learning_rate=FLAGS.learning_rate)

'''random_agent = RandomAgent(width * height * pieces_num, env, -1)
print(play(env, agent1, random_agent, iterations=iterations))

random_agent = RandomAgent(width * height * pieces_num, env, 1)
print(play(env, random_agent, agent2, iterations=iterations))

agent1.set_exploration(1, iterations)
agent2.set_exploration(1, iterations)'''

print(play(env, agent1, agent2, iterations=iterations))
agent1.save("weights1.hdf5")
agent2.save("weights2.hdf5")

'''human_agent = HumanAgent(width * height * pieces_num, env)
play(env, agent1, human_agent, performance_test=False)'''
