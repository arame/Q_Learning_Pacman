import numpy as np
import random
from hyper import Hyper

class Q_learn:
    def __init__(self, no_actions):
        # The state is a combination of cell id and whether it is empty, a breadcrumb or an obstacle.
        # The state of a cell can change from breadcrumb to empty for the same cell id
        # As a result our q table will exist in 3 dimensions; 
        # (state cell_id, state type (ie breadcrumb or empty), actions).
        # We need a dictionary of indexes for each cell. This defaults to zero for each cell.
        # And where a cell with a breadcrumb changes to empty, 
        # the index for that cell changes from 0 to 1.
        # The breadcrumb to empty change is the only scenario where this happens.
        self.no_cells = Hyper.N * Hyper.N
        self.no_actions = no_actions
        self.q_indexes = {i:0 for i in range(self.no_cells)}
        self.Q_table = np.zeros((self.no_cells, 2, no_actions), dtype=np.float)

    def reset(self):
        # By setting all the q indexes to zero, the q table will be set to
        # point to all the breadcrumbs as originally set
        self.q_indexes = {i:0 for i in range(self.no_cells)}

    def update(self, old_cell_id, new_cell_id, action, reward):
        alpha = Hyper.alpha
        gamma = Hyper.gamma
        q_old = self.get_q_value(old_cell_id, action)
        q_max = self.get_max_q(new_cell_id)
        q_val = q_old + alpha * (reward + gamma * q_max - q_old)
        self.set_q_value(old_cell_id, action, q_val)

    # Get and set q values in the table, making sure the q_indexes dictionary is referenced
    # before accessing or changing the Q table
    def get_q_value(self, cell_id, action):
        index = self.q_indexes[cell_id]
        q_val = self.Q_table[cell_id, index, action]
        return q_val

    def set_q_value(self, cell_id, action, q_val):
        index = self.q_indexes[cell_id]
        self.Q_table[cell_id, index, action] = q_val 

    def get_max_q(self, cell_id):
        index = self.q_indexes[cell_id]
        q_max = np.max(self.Q_table[cell_id, index, :]) 
        return q_max 

    def get_actions_for_cell_id(self, cell_id):
        index = self.q_indexes[cell_id]
        actions = self.Q_table[cell_id, index, :]
        return actions

    def get_action_for_max_q(self, cell_id):
        actions = self.get_actions_for_cell_id(cell_id)
        # For greedy policy get the index of the maximum value
        # in the actions array. 
        # If more than 1 index is returned, choose 1 randomly
        _actions = np.where(actions == np.amax(actions))
        _action = np.random.choice(_actions[0], 1).item()
        return _action

    def print(self):
        print("Q table indexes")
        print(self.q_indexes)
        print("Updated Q Table")

        for i in range(self.no_cells):
            print(f"{i}, 0  | {self.Q_table[i, 0, :]}  ///  1  | {self.Q_table[i, 1, :]}")
            if (i + 1) % Hyper.N == 0:
                print("-----------------------------------------------------------------------")