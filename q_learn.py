import numpy as np
from hyper import Hyper
from constants import Constants

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
        self.Q_table = np.zeros((self.no_cells, 2, no_actions), dtype=np.int8)

    def reset(self):
        # By setting all the q indexes to zero, the q table will be set to
        # point to all the breadcrumbs as originally set
        self.q_indexes = {i:0 for i in range(self.no_cells)}

    def update(self, old_state, new_state, action, reward, is_breadcrumb):
        alpha = Hyper.alpha
        gamma = Hyper.gamma
        q_old = self.get_q_value(old_state, action)
        q_max, _ = self.get_max_q(new_state)
        q_new = q_old + alpha * (reward + gamma * q_max - q_old)
        self.set_q_value(new_state, action, q_new, is_breadcrumb)

    # Get and set q values in the table, making sure the q_indexes dictionary is referenced
    # before accessing or changing the Q table
    def get_q_value(self, state, action):
        index = self.q_indexes[state]
        return self.Q_table[state, index, action]

    def set_q_value(self, state, action, q_val, is_breadcrumb):
        index = self.q_indexes[state]
        self.Q_table[state, index, action] = q_val  
        if is_breadcrumb:
            # When the Pacman leaves the breadcrumb cell, it will change state and become empty
            self.q_indexes[state] = 1  
        print("Updated Q Table")
        print("---------------")
        """ for i in range(self.no_cells):
            print(i, "0  |", self.Q_table[i, 0, :])
            print(i, "1  |", self.Q_table[i, 1, :])
            print("---------------") """

    def get_max_q(self, state):
        index = self.q_indexes[state]
        q_max = np.max(self.Q_table[state, index, :]) 
        return q_max, index 

    def get_action_for_max_q(self, state):
        q_max, index = self.get_max_q(state)
        actions = self.Q_table[state, index, :]
        for i in range(len(actions)):
            if actions[i] == q_max:
                return actions[i]
        return actions[0]
