#!/usr/bin/env python3

from main import Board, Agent, load
from time import sleep

Q = load('qtable.t')
rows, columns = 10, 15
a = Agent(0.1, [0, 1, 2, 3])

b = Board(rows, columns, a)
for new_state in range(rows * columns):
    action, exploration = a.get_action(Q, new_state, 0)
    reward, c = b.move(action)
    
    print(chr(27) + "[2J")
    print(b)
    print('Moves:',b.n_moves)

    if c:
        break
    
    sleep(0.1)