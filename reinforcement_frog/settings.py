rows, columns = 10, 15
epsilon = 0.2
board_epsilon = .85
alpha = 0.2
gamma = 0.95
actions = list(range(4))
n_actions = len(actions)

ACTIONS = {
    0: 'UP', 
    1: 'DOWN', 
    2: 'LEFT', 
    3:'RIGHT'
}

EMOJIS = {
    'frog': '\U0001F438',
    'objective': '\U0001F40C',
}