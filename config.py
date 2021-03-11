
class Hyper:
    #total_episodes = 50
    total_episodes = 500
    N = 7
    gamma = 0.99
    alpha = 0.9
    init_epsilon = 1
    decay = 0.9998
    epsilon_threshold = 0.01
    no_breadcrumbs = 10
    is_ghost = True
    show_step = False

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"Threshold for exploitation (epsilon) = {Hyper.init_epsilon}")
        print(f"epsilon decay = {Hyper.decay}")
        print(f"minimum value of epsilon = {Hyper.epsilon_threshold}")
        print(f"learning rate (alpha) = {Hyper.alpha}")
        print(f"discount factor (gamma) = {Hyper.gamma}")
        print(f"total number of breadcrumbs {Hyper.no_breadcrumbs}")

class Constants:
    EMPTY = 0
    BREADCRUMB = 1
    OBSTACLE = 2
    START = 3
    GHOST = 4
    AGENT = 9
    EMPTY_X = "."
    BREADCRUMB_X = "b"
    OBSTACLE_X = "X"
    START_X = "S"
    AGENT_X = "A"
    GHOST_X = "G"
    EMPTY_REWARD = -1
    BREADCRUMB_REWARD = 10
    OBSTACLE_REWARD = -100
    GHOST_REWARD = -500
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    GHOST = 4
    OBSTACLE_COORDS = [(1, 1), (1, 4), (2, 2), (3,4), (4, 2), (4, 4), (5, 4)]
    WIN_CELL = 0
    LOSE_CELL = 1