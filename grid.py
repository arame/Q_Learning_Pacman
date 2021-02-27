import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sn
from policy import Policy
from collections import namedtuple
from hyper import Hyper
from constants import Constants
from q_learn import Q_learn

class Pacman_grid:
    def __init__(self):
        self.no_cells = Hyper.N * Hyper.N
        self.no_episodes = 0
        self.setup_display_dict()
        self.setup_env()
        self.setup_reward_dict()
        self.setup_action_dict()
        self.Q = Q_learn(self.no_actions)
        self.policy = Policy()
        self.timesteps_per_episode = []
        self.rewards_per_episode = []

    def setup_env(self):
        self.state_position_dict = {(i * Hyper.N + j):(i, j) for i in range(Hyper.N) for j in range(Hyper.N)}
        self.position_state_dict = {v: k for k, v in self.state_position_dict.items()}

        self.env = np.zeros((Hyper.N, Hyper.N), dtype = np.int8)
        self.env_counter = np.zeros((Hyper.N, Hyper.N), dtype = np.int16)
        # Borders are obstacles
        self.env[0, :] = self.env[-1, :] = self.env[:, 0] = self.env[:, -1] = Constants.OBSTACLE
        
        # Start cell in the middle
        _, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.START

        # Replace empty cells with obstacles. 
        #no_obstacles = Hyper.N - 2
        self.populate_env_with_obstacles()
        # Replace empty cells with breadcrumbs. 
        self.populate_env_with_breadcrumbs()
        self.orig_env = np.copy(self.env)

    def populate_env_with_obstacles(self):
        # selecting obstacles in random locations risks the possibility of
        # insoluble games with a breadcrumb inaccessible surrounded by obstacles.
        # To rectify this, obstacles are set from a list of coordinates
        for coord in Constants.OBSTACLE_COORDS:
            self.env[coord[0], coord[1]] = Constants.OBSTACLE

    def populate_env_with_breadcrumbs(self):
        # Keep a record of the breadcrumb coordinates
        # This can be used to calculate the index of the Q table
        self.populate_env_with_state(Constants.BREADCRUMB, Hyper.no_breadcrumbs)
        arr_temp = np.nonzero(self.env == Constants.BREADCRUMB)
        self.id_breadcrumb_coords = {i : (arr_temp[0][i], arr_temp[1][i]) for i in range(len(arr_temp[0]))}
        self.breadcrumb_coords_id = {v: k for k, v in self.id_breadcrumb_coords.items()}

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

    def reset(self):
        # reset the breadcrumb indexes on the Q matrix
        self.Q.reset()
        # set the grid to how it was before
        self.env = np.copy(self.orig_env)
        # put agent in the start cell of the environment
        start_cell_id, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.AGENT
        self.agent_cell_id = start_cell_id
        self.time_step = 0
        self.total_reward_per_episode = 0
        self.done = False
        self.breadcrumb_cnt = 0
        self.prev_state = Constants.START
        self.no_episodes += 1

    def step(self):
        # Q Learning algorithm code takes place here
        action = self.policy.get(self.agent_cell_id, self.Q)
        new_cell_id = self.get_cell_id_for_action(action)
        reward = self.get_reward(new_cell_id)
        self.total_reward_per_episode += reward
        self.Q.update(self.agent_cell_id, new_cell_id, action, reward)
        self.agent_step(new_cell_id)
        self.time_step += 1
        if self.time_step > 1000:
            print("Too many timesteps")
            self.done = True
            return self.done

        self.done = self.breadcrumb_cnt == Hyper.no_breadcrumbs
        return self.done

    def check_if_cell_breadcrumb(self, cell_id):
        i, j = self.state_position_dict[cell_id]
        state = self.env[i, j]
        is_breadcrumb = state == Constants.BREADCRUMB
        return is_breadcrumb

    def get_reward(self, cell_id):
        i, j = self.state_position_dict[cell_id]
        state = self.env[i, j]
        reward = self.reward_dict[state]
        return reward

    def agent_step(self, new_cell_id):
        # check if the new cell location is on an obstacle
        # if it is, do not change the environment or move the agent
        i, j = self.state_position_dict[new_cell_id]
        if self.env[i, j] == Constants.OBSTACLE:
            return

        # When the Pacman agent moves from the start cell, 
        # that cell will be empty
        i, j = self.state_position_dict[self.agent_cell_id]
        if self.prev_state == Constants.BREADCRUMB:
            # When the Pacman agent leaves the breadcrumb cell, 
            # it will change state to empty in the grid
            # and become empty in the Q table
            breadcrumb_id = self.breadcrumb_coords_id[i, j]
            self.Q.update_Q_table_index(breadcrumb_id)
            self.breadcrumb_cnt += 1 

        # The Pacman agent leaves the current cell, which will be then empty
        self.env[i, j] = Constants.EMPTY
        
        self.agent_cell_id = new_cell_id
        i, j = self.state_position_dict[self.agent_cell_id]
        self.prev_state = self.env[i, j]
        self.env[i, j] = Constants.AGENT
        self.env_counter[i, j] += 1

    def get_cell_id_for_action(self, action):
        # to move the agent, get the coordinates of the current cell
        # change one of the coordinates, and return the cell_id of the new cell
        i, j = self.state_position_dict[self.agent_cell_id]
        _action = self.index_to_actions[action]
        i += _action.delta_i
        j += _action.delta_j
        new_cell_id = self.position_state_dict[i, j]
        return new_cell_id

    def print_grid(self, caption):
        # Use characters rather than integers to make it easier to interpret the grid
        print(caption)
        lower = 0
        higher = Hyper.N - 1
        for i in range(Hyper.N):
            line = ''
            for j in range(Hyper.N):
                state_id = self.orig_env[i,j]
                line += self.dict_map_display[state_id] + " "
            line += f"    cells {lower} - {higher}"
            print(line)
            lower += Hyper.N
            higher += Hyper.N

    def print_episode_results(self, episodes):
        caption = f"Completed environment after {episodes} episodes and {self.time_step} timesteps, total reward: {self.total_reward_per_episode} with epsilon: {self.policy.epsilon}"
        print(caption)

    def save_episode_stats(self):
        self.timesteps_per_episode.append(self.time_step)
        self.rewards_per_episode.append(self.total_reward_per_episode)

    def print_results(self):
        hm_filename = f"images/hm_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        rw_filename = f"images/rw_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        ts_filename = f"images/ts_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        self.print_grid("Initial Environment")
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        _ = sn.heatmap(data=self.env_counter)
        plt.title("Number of steps per cell")
        plt.xlabel(x_label_text)
        plt.savefig(hm_filename)

        fig = plt.figure()
        fig.add_subplot(111)
        
        episodes = np.arange(1, len(self.timesteps_per_episode)+1)
        plt.title(f"Number of timesteps per episode for {Hyper.no_breadcrumbs} breadcrumbs")
        plt.plot(episodes, self.timesteps_per_episode)
        plt.ylabel('Steps')
        plt.xlabel(x_label_text)
        plt.savefig(rw_filename)

        fig = plt.figure()
        fig.add_subplot(111)
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        episodes = np.arange(1, len(self.rewards_per_episode)+1)
        plt.title(f"Value of rewards per episode for {Hyper.no_breadcrumbs} breadcrumbs")
        plt.plot(episodes, self.rewards_per_episode)
        plt.ylabel('Rewards')
        plt.xlabel(x_label_text)
        plt.savefig(ts_filename)

