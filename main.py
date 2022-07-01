#!/usr/bin/env python3

import random
import enum
import os
import pickle
import numpy as np

from typing import List

Direction = enum.Enum('Direction', 'UP DOWN LEFT RIGHT')

def save(obj, name):
    with open(name, mode='wb') as fp:
        fp.write(pickle.dumps(obj))

class Agent:
    def __init__(self, epsilon: float, actions: List[Direction]) -> None:
        self.pos = (0, 0)
        self.epsilon = epsilon
        self.actions = actions
        self.explorations = 0

    def get_action(self, Q: List):
        i = np.random.random()

        # Explore
        if i <= self.epsilon:
            self.explorations += 1
            return random.choice(self.actions), True

        # Exploit best action
        # return max(enumerate(max(Q)), key=lambda x: x[1])[0] + 1, False

        return self.actions[np.argmax(np.argmax(Q, axis=0))], False

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
        
        self.goal_pos = self.set_goal()
    
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
    
    def reward(self, i=-1, j=-1):
        # euclidean distance
        
        if i != -1 and j != -1:
            r = ((self.goal_pos[0] - i) ** 2 + (self.goal_pos[1] - j) ** 2) ** 0.5
        else:
            r = ((self.goal_pos[0] - self.agent.pos[0]) ** 2 + (self.goal_pos[1] - self.agent.pos[1]) ** 2) ** 0.5

        return self.rows - r

    def get_pos(self, direction: Direction):
        i, j = self.agent.pos

        if direction == Direction.UP.value:
            i += 1
        elif direction == Direction.DOWN.value:
            i -= 1
        elif direction == Direction.LEFT.value:
            j -= 1
        elif direction == Direction.RIGHT.value:
            j += 1
        
        if j <= 0:
            j = 0
        
        if i <= 0:
            i = 0
        
        if j > self.columns-1:
            j -= 1
        
        if i > self.rows-1:
            i -= 1

        return i, j


    def move(self, direction: Direction):
        i, j = self.agent.pos
        self.board[i][j] = '.'

        i, j = self.get_pos(direction)

        if (i, j) == self.goal_pos and self.board[i][j] == self.fruit and not self.completed:
            self.completed = True
            return self.reward() * 4, self.completed

        self.board[i][j] = self.agent
        self.agent.pos = (i, j)

        return self.reward(), self.completed

rows, columns = 7, 7
gamma = 0.95
alpha = 0.15
epsilon = 0.1
n_actions = len(Direction)
Q = np.zeros((rows*columns, n_actions))
actions = [e.value for e in Direction]

try:
    gen = 1
    completed = 0
    explorations = 0

    while True:
        state = 0
        a = Agent(epsilon, actions)
        b = Board(rows, columns, a)

        for new_state in range(rows * columns):
            os.system('clear')
            action, exploration = a.get_action(Q)
            reward, c = b.move(action)

            if exploration:
                explorations += 1
            
            if c:
                completed += 1
                break
            
            ai = actions.index(action)
            qmax = np.max(Q[new_state, :])
            q = Q[state, ai] + alpha * (reward + gamma * qmax - Q[state, ai])
            Q[state, ai] = q

            print(b)
            print('Generation', gen)
            print('Completed', completed)
            print('Explorations', explorations)
            print('Accuracy', completed / gen)

            state = new_state
        gen +=1
except KeyboardInterrupt:
    save(Q, 'qtable.t')