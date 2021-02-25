import numpy as np
from policy import Policy
from collections import namedtuple
from hyper import Hyper
from constants import Constants
from q_learn import Q_learn

class Pacman_grid:
    def __init__(self):
        self.no_cells = Hyper.N * Hyper.N
        self.setup_display_dict()
        self.setup_env()
        self.setup_reward_dict()
        self.setup_action_dict()
        self.Q = Q_learn(self.no_actions)
        self.policy = Policy()

    def setup_env(self):
        self.state_position_dict = {(i * Hyper.N + j):(i, j) for i in range(Hyper.N) for j in range(Hyper.N)}
        self.position_state_dict = {v: k for k, v in self.state_position_dict.items()}

        self.env = np.zeros((Hyper.N, Hyper.N), dtype = np.int8)
        # Borders are obstacles
        self.env[0, :] = self.env[-1, :] = self.env[:, 0] = self.env[:, -1] = Constants.OBSTACLE
        
        # Start cell in the middle
        _, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.START

        # Replace empty cells with obstacles. 
        no_obstacles = Hyper.N - 2
        self.populate_env_with_state(Constants.OBSTACLE, no_obstacles)

        # Replace empty cells with breadcrumbs. 
        self.no_breadcrumbs = (Hyper.N - 2) * 2
        self.populate_env_with_state(Constants.BREADCRUMB, self.no_breadcrumbs)
        self.orig_env = np.copy(self.env)
        self.print_grid("Initial Environment")

    def populate_env_with_state(self, state, limit):
        # This method is needed as sometimes the empty cells returned have duplicates
        count = 0
        while count < limit:
            no_cells = limit - count
            empty_coord = self.get_empty_cells(no_cells)
            self.env[empty_coord[0], empty_coord[1]] = state
            count = np.count_nonzero(self.env == state)

    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.env == Constants.EMPTY)
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates

    def get_start_cell_coords(self):
        # Start cell in the middle
        start_cell_id = int((self.no_cells - 1) / 2)     # N must be an odd number
        i, j = self.state_position_dict[start_cell_id]
        return start_cell_id, i, j

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
        self.no_actions = len(self.index_to_actions)

    def setup_display_dict(self):
        self.dict_map_display={ Constants.EMPTY: Constants.EMPTY_X,
                                Constants.BREADCRUMB: Constants.BREADCRUMB_X,
                                Constants.OBSTACLE: Constants.OBSTACLE_X,
                                Constants.START: Constants.START_X,
                                Constants.AGENT: Constants.AGENT_X}

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
        self.Q.reset()
        # set the grid to how it was before
        self.env = np.copy(self.orig_env)
        # put agent in the start cell of the environment
        start_cell_id, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.AGENT
        self.agent_cell_id = start_cell_id
        self.time_step = 0
        self.result = ""
        self.is_breadcrumb = False
        self.done = False
        self.breadcrumb_cnt = 0

    def step(self):
        # Q Learning algorithm code takes place here
        action = self.policy.get(self.agent_cell_id, self.Q)
        self.policy.update_epsilon()
        new_cell_id = self.agent_step(action)
        reward = self.get_reward(new_cell_id)
        self.Q.update(self.agent_cell_id, new_cell_id, action, reward, self.is_breadcrumb)
        self.move_agent(new_cell_id)
        self.time_step += 1
        self.print_grid(f"Next Step {self.time_step}: {self.index_to_actions[action].name}")
        if self.is_breadcrumb:
            self.breadcrumb_cnt += 1
            self.done = self.breadcrumb_cnt == self.no_breadcrumbs

        return self.done

    def get_reward(self, cell_id):
        i, j = self.state_position_dict[cell_id]
        state = self.env[i, j]
        self.is_breadcrumb = state == Constants.BREADCRUMB
        reward = self.reward_dict[state]
        return reward

    def move_agent(self, new_cell_id):
        # check if the new cell location is on an obstacle
        # if it is, do not change the environment or move the agent
        i, j = self.state_position_dict[new_cell_id]
        if self.env[i, j] == Constants.OBSTACLE:
            return
        # When an agent moves from a cell, that cell will be empty
        # This will overwrite the previous state,
        # which might be Start, breadcrumb or empty
        i, j = self.state_position_dict[self.agent_cell_id]
        self.env[i, j] = Constants.EMPTY
        self.agent_cell_id = new_cell_id
        i, j = self.state_position_dict[self.agent_cell_id]
        self.env[i, j] = Constants.AGENT

    def agent_step(self, action):
        # to move the agent, get the coordinates of the current cell
        # change one of the coordinates, and return the cell_id of the new cell
        i, j = self.state_position_dict[self.agent_cell_id]
        _action = self.index_to_actions[action]
        i += _action.delta_i
        j += _action.delta_j
        new_cell_id = self.position_state_dict[i, j]
        return new_cell_id