import numpy as np
from hyper import Hyper

class Q_learn:
    def __init__(self, no_cells, no_actions):
        # The state is a combination of cell id and whether it is empty, a breadcrumb or an obstacle.
        # The state of a cell can change from breadcrumb to empty for the same cell id
        # As a result our q table will exist in 3 dimensions; 
        # (state cell_id, state type (ie breadcrumb or empty), actions).
        # We need a dictionary of indexes for each cell. This defaults to zero for each cell.
        # And where a cell with a breadcrumb changes to empty, 
        # the index for that cell changes from 0 to 1.
        # The breadcrumb to empty change is the only scenario where this happens.
        self.no_cells = no_cells
        self.no_actions = no_actions
        self.q_indexes = {i:0 for i in range(self.no_cells)}
        self.Q_table = np.zeros((self.no_cells, 2, no_actions), dtype=np.int8)

    def reset(self):
        # By setting all the q indexes to zero, the q table will be set to
        # point to all the breadcrumbs as originally set
        self.q_indexes = {i:0 for i in range(self.no_cells)}

    def update(self, new_state, new_action, old_state, old_action, reward, is_breadcrumb):
        alpha = Hyper.alpha
        gamma = Hyper.gamma
        old_index = self.q_indexes[old_state]
        q_old = self.Q_table[old_state, old_index, old_action]
        q_max = self.calcMaxQ()
        q_new = q_old + alpha * (reward + gamma * q_max - q_old)
        self.set_q_value(state, action, q_val, is_breadcrumb)

    def calcMaxQ(self):
        q_max = 0
        for s in range(self.no_cells):
            for a in range(self.no_actions):
                q_val = get_q_value(self, s, a)
                if q_max < q_val:
                    q_max = q_val
        return q_max

    # Get and set q values in the table, making sure the 
    def get_q_value(self, state, action):
        index = self.q_indexes[state]
        return self.Q_table[state, index, action]

    def set_q_value(self, state, action, q_val, is_breadcrumb):
        index = self.q_indexes[state]
        self.Q_table[state, index, action] = q_val  
        if is_breadcrumb:
            # When the Pacman leaves the breadcrumb cell, it will change state and become empty
            self.q_indexes[state] = 1  

