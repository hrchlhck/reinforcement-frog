#!/usr/bin/env python3

import pickle
import numpy as np

from settings import (
    rows, columns, 
    gamma, alpha, 
    epsilon, n_actions, 
    actions, board_epsilon,
    ACTIONS
)

from board import Agent, Board
import matplotlib.pyplot as plt

def save(obj, name):
    with open(name, mode='wb') as fp:
        fp.write(pickle.dumps(obj))
    
def load(name):
    with open(name, 'rb') as fp:
        return pickle.loads(fp.read())

if __name__ == '__main__':
    Q = np.zeros((rows * columns, n_actions))
    bestQ = None

    total_mean_score = []
    completed = 0
    explorations = 0
    max_reward = -30000
    max_gen = 3000
    last = 1
    for gen in range(1, max_gen + 1):
        state = 0
        a = Agent(epsilon, actions, max_gen // 2)
        b = Board(rows, columns, a, board_epsilon, True)

        scores = []
        for new_state in range(rows * columns):
            action, exploration = a.get_action(Q, b.board, state, gen-1)
            reward, dist, c = b.move(action)

            if exploration:
                explorations += 1
            
            if c:
                completed += 1
                break
            
            if reward > max_reward:
                max_reward = reward
                bestQ = Q.copy()

            scores.append(reward)
            ai = actions.index(action)
            qmax = np.max(Q[new_state, :])
            Q[state, ai] += alpha * (reward + gamma * qmax - Q[state, ai])

            s = f"""
            Generation\t{gen}
            Completed\t{completed}
            Explorations\t{explorations}
            Accuracy\t{completed / gen * 100}
            Moves\t{b.n_moves}
            Avg. Score\t{np.nanmean(total_mean_score)}
            Score\t{reward}
            Max. Reward\t{max_reward}
            Action:\t{ACTIONS.get(action)}
            Distance:\t{dist}
            """
            print(chr(27) + "[2J")
            print(s, end='\r')
            state = new_state

        total_mean_score.append(np.nanmean(scores))

        last = gen

    save(bestQ, 'best_qtable.t')
    save(Q, 'qtable.t')
