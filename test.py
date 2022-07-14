#!/usr/bin/env python3

from settings import (
    rows, columns, 
    epsilon, actions,
    board_epsilon
)
from main import Board, Agent, load
from time import sleep

Q = load('qtable.t')
n_test = 10
n_completed = 0
for i in range(n_test):
    a = Agent(epsilon, actions, 1500)
    b = Board(rows, columns, a, board_epsilon, random_agent_pos=False)
    for new_state in range(rows * columns):
        action, exploration = a.get_action(Q, b.board, new_state, 0)
        reward, dist, c = b.move(action)

        print(chr(27) + "[2J")
        print(b)
        print('Moves:',b.n_moves)
        print('Test n:', i+1)
        print('Completed:', n_completed)

        if c:
            n_completed += 1
            sleep(2)
            break
        
        sleep(0.07)
