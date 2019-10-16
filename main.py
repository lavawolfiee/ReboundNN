from ReboundEnv import *
from NNAgent import *
from RandomAgent import *
from HumanAgent import *


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
            print('Iter ' + str(i))

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
            _wins, draws = test_performance(env, agent1, 1)
            print("Performance of agent1: " + str(_wins) + "% (draws: " + str(draws) + "%)")
            _wins, draws = test_performance(env, agent2, -1)
            print("Performance of agent2: " + str(_wins) + "% (draws: " + str(draws) + "%)")
    return wins


def test_performance(env, agent, turn):
    random_agent = RandomAgent(width * height * pieces_num, env, turn * -1)
    agents = [agent, random_agent]
    if turn == -1:
        agents = agents[::-1]

    wins = play(env, agents[0], agents[1], iterations=100, train=False, performance_test=False)
    games = sum(wins)
    return wins[turn + 1]*100/games, wins[1]*100/games


width = 3
height = 3
pieces_count = 2
pieces_num = 2
iterations = 100

env = ReboundEnv(width, height, pieces_count=pieces_count)
agent1 = NNAgent(width * height, width * height * pieces_num, env, 1,
                 iterations=iterations, actions_in_iter=pieces_count * 2)
agent2 = NNAgent(width * height, width * height * pieces_num, env, -1,
                 iterations=iterations, actions_in_iter=pieces_count * 2)

print(play(env, agent1, agent2, iterations=iterations))

human_agent = HumanAgent(width * height * pieces_num, env)
play(env, agent1, human_agent, performance_test=False)

'''for i in range(iterations):
    env.reset()
    state = env.get_state()
    done = False
    flag = False
    rewards = [0, 0]
    j = 0
    done_counter = 2
    print('Iter ' + str(i))

    while done_counter > 0:
        if done:
            done_counter -= 1

        #print("Scores: " + str(env.scores))
        #print("Pieces: " + str(env.pieces))
        #print(env)
        #print(state)
        # env.render()
        action = agents[j].choose_action(state, rewards[j], done)
        if not done:
            state, rewards[j], done = env.step(action)

        if done and not flag:
            flag = True
            rewards[0] += env.get_end_reward(1)
            rewards[1] += env.get_end_reward(-1)

        agents[j].update()

        j = (j + 1) % 2

    # print("End of game. Winner: " + str(env.winner) + "\n")
    # for a in agents:
    #    a.replay()

while True:
    print("-"*20)
    env.reset()
    a = int(input("\nChoose your opponent (1 or 2): "))
    a -= 1
    a = min(max(a, 0), 1)
    done = False
    j = 0
    state = env.get_state()

    while not done:
        if j == a:
            print("NN turn")
            action = agents[a].choose_action(state, 0, done, train=False)
            state, reward, done = env.step(action)
            agents[a].update(train=False)
            print("Action: " + str(action) + "\n")
        else:


            action =
            state, reward, done = env.step(action)
            print()

        if done:
            env.check_winner()
            print("End of game. Winner: " + str(env.winner) + "\n\n")

        j = (j + 1) % 2'''
