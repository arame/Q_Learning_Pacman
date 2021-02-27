
class Hyper:
    #total_episodes = 50
    total_episodes = 500
    N = 7
    gamma = 0.99
    alpha = 0.1
    init_epsilon = 1
    decay = 0.995
    epsilon_threshold = 0.1
    no_breadcrumbs = 10

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

    