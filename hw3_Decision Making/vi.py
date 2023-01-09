#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function


def vi(env: np.array, goal: Tuple) -> (np.array, np.array):
    """
    env is the grid enviroment
    goal is the goal state
    """
    policy, cost_to_go = np.zeros(env.shape, 'b'), np.ones(env.shape) * 1e2
    cost_to_go[goal] = 0
    while True:
        is_new = False
        for i in range(len(cost_to_go)):
            for j in range(len(cost_to_go)):
              index = [i,j]
              for act_idx, action in enumerate(action_space):
                  idx, no_collise = transition_function(env, index , action)
                  if no_collise:
                      new_cost = 1 + cost_to_go[idx]
                      if new_cost < cost_to_go[i,j]:
                          is_new = True
                          cost_to_go[i,j] = new_cost
                          policy[i,j] = act_idx
        assert cost_to_go[goal] == 0
        if not is_new:
            break
    return policy, cost_to_go
