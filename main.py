#!/usr/bin/env python3

import os
import pickle
import numpy as np

from typing import List

import matplotlib.pyplot as plt

def save(obj, name):
    with open(name, mode='wb') as fp:
        fp.write(pickle.dumps(obj))
    
def load(name):
    with open(name, 'rb') as fp:
        return pickle.loads(fp.read())

class Agent:
    def __init__(self, epsilon: float, actions: list) -> None:
        self.pos = (0, 0)
        self.epsilon = epsilon
        self.actions = actions
        self.explorations = 0

    def get_action(self, Q: List, state, limit):
        i = np.random.uniform(0, 1)

        # Explore
        if i <= self.epsilon and limit < 1500:
            self.explorations += 1
            return np.random.choice(self.actions), True

        # Exploit best action
        # return max(enumerate(max(Q)), key=lambda x: x[1])[0] + 1, False
        return self.actions[np.argmax(Q[state, :])], False

    def __str__(self) -> str:
        return '\U0001F438'

class Board:
    def __init__(self, rows: int, columns: int, agent: Agent) -> None:
        self.rows = rows
        self.columns = columns
        self.board = [[' ' for _ in range(columns)] for _ in range(rows)]
        self.board[0][0] = agent
        self.agent = agent
        self.completed = False
        self.fruit = '\U0001F40C'
        self.n_moves = 0
        self.desired_moves = (rows ** 2 + columns ** 2) ** 0.5
        self.goal_pos = self.set_goal()
        self.last_move = None
    
    def __str__(self) -> str:
        ret = ''
        for row in self.board:
            line = ''
            for column in row:
                line += '│ ' + str(column) + ' '

            line += '│'
            delim = '├' + ('─' * (len(line)-2)) + '┤' + '\n'
            ret += line + '\n'
            ret += delim
            ret1 = delim + ret
        return ret1
    
    def set_goal(self, epsilon=0):
        i, j = self.rows-1, self.columns-1

        if np.random.random() <= epsilon:
            i = np.random.randint(low=0, high=self.rows-1)
            j = np.random.randint(low=0, high=self.columns-1)

        if i == 0 and j == 0:
            print('Resetting goal')
            return self.set_goal()

        self.board[i][j] = self.fruit
        return i, j
    
    def reward(self, current_direction, oldi, oldj):
        # euclidean distance
        r = ((self.goal_pos[0] - self.agent.pos[0]) ** 2 + (self.goal_pos[1] - self.agent.pos[1]) ** 2) ** 0.5
        
        if self.desired_moves > self.n_moves > self.desired_moves * 2:
            return 30

        if self.desired_moves > self.n_moves > self.desired_moves * 2 and r < 4:
            return -r - 15

        if self.n_moves <= self.desired_moves * 2 or self.agent.pos == self.goal_pos:
            return 500

        if r < 2:
            return 100

        if self.last_move == current_direction or self.board[oldi][oldj] == '.':
            return -r - 5

        if r > 0:
            return -r
        
        return r

    def get_pos(self, direction: int):
        i, j = self.agent.pos

        # up
        if direction == 0:
            i += 1
        
        # down
        elif direction == 1:
            i -= 1
        
        # left
        elif direction == 2:
            j -= 1

        # right
        elif direction == 3:
            j += 1
        
        if j < 0:
            j = 0
        
        if i < 0:
            i = 0
        
        if j > self.columns-1:
            j -= 1
        
        if i > self.rows-1:
            i -= 1

        return i, j


    def move(self, direction: int):
        oldi, oldj = self.agent.pos
        self.board[oldi][oldj] = '.'

        self.last_move = direction
        i, j = self.get_pos(direction)

        self.n_moves += 1
        
        if (i, j) == self.goal_pos and self.board[i][j] == self.fruit and not self.completed:
            self.completed = True

        self.board[i][j] = self.agent
        self.agent.pos = (i, j)

        return self.reward(direction, oldi ,oldj), self.completed

if __name__ == '__main__':
    rows, columns = 10, 15
    gamma = 0.8
    alpha = 0.2
    epsilon = 0.2
    actions = list(range(4))
    n_actions = len(actions)
    Q = np.zeros((rows*columns, n_actions))
    bestQ = None

    avg_score_gen = []
    completed = 0
    explorations = 0
    max_reward = -30000
    max_gen = 3000
    last = 1
    for gen in range(1, max_gen+1):
        state = 0
        a = Agent(epsilon, actions)
        b = Board(rows, columns, a)

        scores = []
        for new_state in range(rows * columns):
            action, exploration = a.get_action(Q, new_state, gen-1)
            reward, c = b.move(action)

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

            # os.system('clear')
            # print(b, flush=True)
            s = f"""
            Generation\t{gen}
            Completed\t{completed}
            Explorations\t{explorations}
            Accuracy\t{completed / gen * 100}
            Moves\t{b.n_moves}
            Avg. Score\t{np.mean(avg_score_gen)}
            Score\t{reward}
            """
            print(chr(27) + "[2J")
            print(s, end='\r')
            state = new_state

        if not len(avg_score_gen):
            avg_score_gen.append(np.mean(scores))
        else:
            avg_score_gen.append((np.mean(scores) + avg_score_gen[last-1])/2)
        last = gen


        # if gen % 50 == 0 and gen > 0:  
        #     plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
        #     plt.plot([i for i in range(len(avg_score_gen))], avg_score_gen)
        #     plt.xlabel('Generation')
        #     plt.ylabel('Average Score')
        #     plt.savefig('teste.png')

    save(bestQ, 'best_qtable.t')
    save(Q, 'qtable.t')