
import random
import numpy as np
from hyper import Hyper

class Policy():
    
    def __init__(self):
        self.epsilon = Hyper.init_epsilon
        
    def get(self, state, q_values):
        # Sample an action from the policy, given a state
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            cell_idx = self.state_idx_dict[state]
            index_action = np.argmax(q_values[state])
        else:
            index_action = random.randint(0, 3)
        
        action = self.action_dict[index_action]
        return action
        
    def update_epsilon(self):
        # call for each episode
        self.epsilon *= Hyper.decay
        return self.epsilon
        