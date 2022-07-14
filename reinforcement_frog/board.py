from .agent import Agent
from .settings import EMOJIS
import numpy as np

class Board:
    def __init__(self, rows: int, columns: int, agent: Agent, goal_randomness: float, random_agent_pos=False, emoji='objective') -> None:
        self.rows = rows
        self.columns = columns
        self.board = np.array([[' ' for _ in range(columns)] for _ in range(rows)])
        self.agent = agent
        self.completed = False
        self.fruit = EMOJIS.get(emoji)
        self.n_moves = 0
        self.desired_moves = (rows ** 2 + columns ** 2) ** 0.5
        self.last_move = None
        self.goal_pos = self.set_goal(epsilon=goal_randomness)
        self.position_agent(random=random_agent_pos)
    
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
    
    def get_random_pos(self):
        i = np.random.randint(low=0, high=self.rows-1)
        j = np.random.randint(low=0, high=self.columns-1)
        return i, j

    def position_agent(self, random=False):
        i, j = 0, 0

        if random:
            i, j = self.get_random_pos()
        
        if self.goal_pos == (i, j):
            self.position_agent()
        
        self.agent.pos = (i, j)
        self.board[i][j] = self.agent

    def set_goal(self, epsilon=.1):
        i, j = self.rows-1, self.columns-1

        if np.random.random() <= epsilon:
            i, j = self.get_random_pos()

        if i == 0 and j == 0:
            print('Resetting goal')
            return self.set_goal()

        self.board[i][j] = self.fruit
        return i, j
    
    def reward(self):
        # euclidean distance
        distance = ((self.goal_pos[0] - self.agent.pos[0]) ** 2 + (self.goal_pos[1] - self.agent.pos[1]) ** 2) ** 0.5

        if distance == 0:
            return self.rows * self.columns / self.n_moves, distance

        if distance <= 4:
            return self.rows * self.columns / self.n_moves / 4, distance

        return (-distance / self.columns * self.rows) * self.n_moves, distance

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

        if i < 0:
            i = 0
        elif j < 0:
            j = 0
        
        if i >= self.rows:
            i -= 1
        elif j >= self.columns:
            j -= 1

        return i, j


    def move(self, direction: int):
        oldi, oldj = self.agent.pos
        self.board[oldi][oldj] = '.'

        self.last_move = direction
        p = self.get_pos(direction)
        i, j = p

        self.n_moves += 1

        if p == self.goal_pos and self.board[i][j] == self.fruit and not self.completed:
            self.completed = True

        self.agent.pos = p
        self.board[i][j] = self.agent
        r, dist = self.reward()
        return r, dist, self.completed