
import random
import numpy as np
from hyper import Hyper
from constants import Constants
class Policy():
    
    def __init__(self):
        self.epsilon = Hyper.init_epsilon
        
    def get(self, state, Q):
        # Sample an action from the policy, given a state
        # The action returned here is the numerical representation
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            action = Q.get_action_for_max_q(state)
        else:
            action = random.randint(0, 3)
        
        return action
        
    def update_epsilon(self):
        # called for each episode
        self.epsilon *= Hyper.decay
        return self.epsilon
        