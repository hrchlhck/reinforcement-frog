from typing import List, Tuple

import numpy as np

from reinforcement_frog.settings import EMOJIS

def __default_policy(
        pos: tuple, 
        actions: List[int], 
        state: np.array, 
        epsilon: float, 
        limit: int, 
        exploration_limit: int, 
        Q: List[int], 
        state_index: int
    ):
    x, y = state.shape

    if pos == (x - 1, y - 1):
        actions = [0, 2]
    elif pos == (0, y - 1):
        actions = [0, 1, 3]
    elif pos == (x - 1, 0):
        actions = [0, 1, 2]
    elif pos == (0, 0):
        actions = [1, 2, 3]

    i = np.random.uniform(0, 1)

    # Explore
    if i <= epsilon and limit < exploration_limit:
        return np.random.choice(actions), True

    # Exploit best action
    _state = Q[state_index, :]
    _state = _state[actions]
    return actions[np.argmax(_state)], False

def __mlp_based():
    pass

POLICIES = {
    'default': __default_policy,
}

class Agent:
    def __init__(self, epsilon: float, actions: list, exploration_limit: int, emoji: str='', policy: str = '') -> None:
        self.pos = (0, 0)
        self.epsilon = epsilon
        self.actions = actions
        self.explorations = 0
        self.exploration_limit = exploration_limit

        if not policy:
            self._policy = 'default'
        else:
            self._policy = policy

        if not emoji:
            self._emoji = EMOJIS.get('frog')

    def __str__(self) -> str:
        return self._emoji
    
    def action(self, Q: List[int], state, state_index: int, generation: int) -> Tuple[int, bool]:
        policy = POLICIES.get(self._policy)
        return policy(
            self.pos, 
            self.actions, 
            state,
            self.epsilon,
            generation,
            self.exploration_limit,
            Q,
            state_index
        )

