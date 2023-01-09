#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function, pursuer_transition
from collections import defaultdict
from tqdm import tqdm
import math


def mcts(env: np.array, x_e: Tuple, x_p: Tuple, goal: Tuple, k_budget, default_policy) -> Tuple:
    """
    Monte-Carlo tree search
    env is the grid enviroment
    x_e evader
    x_p pursuer
    goal is the goal state
    """
    Tree = defaultdict(list)
    root = State(x_e, x_p,'evader', goal, False)
    discovered_nodes = [root.get_location()]
    expand(env, root, Tree, discovered_nodes)
    for child in Tree[root]:
        if child.is_terminal:
            return child.get_action(root)
    pbar = tqdm(total=k, desc='inner_loop', leave=False)
    while k > 0:
        node = root
        while not unexplored_node(env, node, Tree):
            node = child_select(node,Tree)
        if node.visited != 0 or is_root(node) : # or not node.is_simulated i guess
            expand(env, node, Tree, discovered_nodes)
            not_explored_children = [child for child in Tree[node] if child.visited == 0]
            if len(not_explored_children) != 0:
                node = np.random.choice(not_explored_children)
        reward = re_simulation(node, default_policy)
        node.is_simulated = True
        backpropagate(node, reward)
        k-=1
        pbar.update(1)
    u = best_child(root,Tree).get_action(root)
    return u

class State:
    def __init__(self,x_e,x_p,turn,goal,is_expanded=False,is_simulated=False, parent=None):
        self.evader = x_e
        self.pursuer = x_p
        self.turn = turn
        self.goal = goal
        self.is_expanded = is_expanded
        self.is_simulated = is_simulated
        self.visited = 0
        self.reward = 0
        self.parent = parent
        self.children = 0
        self.is_terminal = self.evader == self.goal
    def __str__(self):
        return f"""evader={self.evader}, pursuer={self.pursuer}, turn={self.turn},
        {self.is_expanded =},{self.is_simulated =}, {self.visited =}, reward={self.reward}"""
    def get_action(self, root):
        action = tuple(np.array(self.evader)-np.array(root.evader))
        assert action in action_space, f'{action}'
        return action
    def get_location(self):
        return [self.evader,self.pursuer,self.turn]

def unexplored_node(env, node, Tree):
    if len(Tree[node]) == 0:
        return True
    for child in Tree[node]:
        if child.is_simulated == False:
            return True
    return False

def get_children(env,state,discovered_nodes):
        assert state.children == 0
        if state.evader == state.goal:
            return []
        children =[]
        for action in action_space:
            if state.turn == 'evader':
                node = state.evader
            elif state.turn == 'pursuer':
                node = state.pursuer
            may_child, is_child = transition_function(env, node, action)
            if is_child:
                if state.turn == 'evader':
                    child_state = State(may_child,state.pursuer,'pursuer',state.goal,parent=state)
                elif state.turn == 'pursuer':
                    child_state = State(state.evader,may_child,'evader',state.goal,parent=state)
                if child_state.get_location() not in discovered_nodes:
                    children.append(child_state)
        state.children = len(children)
        return children

def expand(env, node, Tree, discovered_nodes):
    if not node.is_expanded:
        children = get_children(env, node, discovered_nodes)
        Tree[node] = children
        discovered_nodes.extend([child.get_location() for child in children])
        node.is_expanded = True

def child_select(node, Tree, weight=2**0.5):
        children = Tree[node]
        assert all(child.is_simulated for child in children)
        vertex = math.log(node.visited)
        def bound(child):
                return child.reward / child.visited + weight * (vertex / child.visited)**0.5
        return max(children, key=bound)

def simulation(x_e=x_e, x_p=x_p, turn:str='evader', policy=default_policy):
    accumulated_reward = 0
    for s in range(100):
        distance_to_goal = np.linalg.norm(np.array(x_e) - np.array(goal))
        if distance_to_goal != 0:
            accumulated_reward += 0.1/distance_to_goal
        else:
            accumulated_reward += 100
        # according to the optimal policy of the evader, move the evader
        u_e = action_space[policy[x_e]]  # default_policy without taking into account the pursuer
        if not (s == 0 and turn=='pursuer'):
            x_e, _ = transition_function(environment, x_e, u_e)
        
        if x_e == goal:
            return 100 + accumulated_reward - (s+1)

        # propagate the pursuer: TODO uncomment the next line to release the beast
        x_p = pursuer_transition(environment,x_e, x_p)
        if x_p == x_e:
            return 0
    return accumulated_reward

def re_simulation(node, policy):
    return simulation(node.evader, node.pursuer, node.turn, policy)

def is_root(node):
    return node.parent is None

def backpropagate(node, reward):
    node.visited += 1
    node.reward += reward
    if node.parent is None: return
    backpropagate(node.parent, reward)

def best_child(node, Tree):
    children = Tree[node]
    def visits(child):
        return child.visited
    return max(children.visited)

def plot_explored(env, Tree, node):
    current_env = np.copy(env)
    for node_ in Tree.keys():
        children = Tree[node_]
        evaders = [child.evader for child in children]+[node_.evader]
        pursuers = [child.pursuer for child in children]+[node_.pursuer]
        goal = node_.goal
        for evader in evaders:
            current_env[evader] = 0.9  # yellow
        current_env[node.evader] = 1
        # plot pursuer
        for pursuer in pursuers:
            current_env[pursuer] = 0.6  # cyan-ish
        current_env[node.pursuer] = 0.4  # cyan-ish
        # plot goal
        current_env[goal] = 0.3
    plt.close()
    fig = plt.figure()
    ax = fig.gca()
    ax.pcolormesh(current_env, edgecolors='k', linewidth=1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()