import numpy as np
from policy import Policy
from collections import namedtuple
from hyper import Hyper
from constants import Constants

class Pacman_grid:
    def __init__(self):
        self.no_cells = Hyper.N * Hyper.N
        self.setup_display_dict()
        self.setup_env()
        self.setup_reward_dict()
        self.setup_action_dict()
        self.setup_q_table()

    def setup_env(self):
        self.state_position_dict = {(i * Hyper.N + j):(i, j) for i in range(Hyper.N) for j in range(Hyper.N)}
        self.position_state_dict = {v: k for k, v in self.state_position_dict.items()}

        self.env = np.zeros((Hyper.N, Hyper.N), dtype = np.int8)
        # Borders are obstacles
        self.env[0, :] = self.env[-1, :] = self.env[:, 0] = self.env[:, -1] = Constants.OBSTACLE
        
        # Start cell in the middle
        start_cell_id = int((self.no_cells - 1) / 2)     # N must be an odd number
        i, j = self.state_position_dict[start_cell_id]
        self.env[i, j] = Constants.START

        # Replace empty cells with obstacles. 
        no_obstacles = Hyper.N - 2
        empty_coord = self.get_empty_cells(no_obstacles)
        self.env[empty_coord[0], empty_coord[1]] = Constants.OBSTACLE

        # Replace empty cells with breadcrumbs. 
        no_breadcrumbs = (Hyper.N - 2) * 2
        empty_coord = self.get_empty_cells(no_breadcrumbs)
        self.env[empty_coord[0], empty_coord[1]] = Constants.BREADCRUMB
        self.orig_env = np.copy(self.env)
        self.print_grid("Initial Environment")
        
    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.env == Constants.EMPTY)
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates

    def setup_reward_dict(self):
        self.reward_dict = {
            Constants.EMPTY: Constants.EMPTY_REWARD,
            Constants.BREADCRUMB: Constants.BREADCRUMB_REWARD,
            Constants.OBSTACLE: Constants.OBSTACLE_REWARD,
            Constants.START: Constants.EMPTY_REWARD
            }

    def setup_action_dict(self):
        _Action = namedtuple('Action', 'name index delta_i delta_j')
        up = _Action('up', 0, -1, 0)    
        down = _Action('down', 1, 1, 0)    
        left = _Action('left', 2, 0, -1)    
        right = _Action('right', 3, 0, 1) 
        self.index_to_actions = {}
        for action in [up, down, left, right]:
            self.index_to_actions[action.index] = action

        self.actions_to_index = {v: k for k, v in self.index_to_actions.items()}

    def setup_q_table(self):
        # The state is a combination of cell id and whether it is empty, a breadcrumb or an obstacle.
        # The state of a cell can change from breadcrumb to empty for the same cell id
        # As a result our q table will exist in 3 dimensions; state x state x actions
        # We need a dictionary of indexes for each cell. This defaults to zero for each cell.
        # And where a breadcrumb changes to empty, the index for that cell changes from 0 to 1.
        # The breadcrumb to empty change is the only scenario where this happens.
        self.q_indexes = {i:0 for i in range(self.no_cells)}
        no_actions = len(self.index_to_actions)
        self.q_table = np.zeros((self.no_cells, 2, no_actions), dtype=np.int8)

    def setup_display_dict(self):
        self.dict_map_display={ Constants.EMPTY: Constants.EMPTY_X,
                                Constants.BREADCRUMB: Constants.BREADCRUMB_X,
                                Constants.OBSTACLE: Constants.OBSTACLE_X,
                                Constants.START: Constants.START_X}

    def print_grid(self, caption):
        # Use characters rather than integers to make it easier to interpret the grid
        print(caption)
        for i in range(Hyper.N):
            line = ''
            for j in range(Hyper.N):
                state_id = self.env[i,j]
                line += self.dict_map_display[state_id] + " "
            print(line)

    def print_episode_results(self, episodes):
        print(f"For episode {episodes}, which {self.result} after {self.time_step} steps")
        caption = "Completed environment"
        self.print_grid(caption)

    def reset(self):
        # By setting all the q indexes to zero, the q table will be set to
        # point to all the breadcrumbs as originally set
        self.q_indexes = {i:0 for i in range(self.no_cells)}
        # set the grid to how it was before
        self.env = np.copy(self.orig_env)
        self.time_step = 0
        self.result = ""

    def step(self):
        # Q Learning algorithm code takes place here
        self.time_step += 1
        return True