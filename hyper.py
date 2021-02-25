
class Hyper:
    #total_episodes = 50
    total_episodes = 1
    N = 7
    gamma = 0.9
    init_epsilon = 0.999
    decay = 0.99
    alpha = 0.5 

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"Threshold for exploitation (epsilon) = {Hyper.init_epsilon}")
        print(f"epsilon decay = {Hyper.decay}")
        print(f"learning rate (alpha) = {Hyper.alpha}")
        print(f"discount factor (gamma) = {Hyper.gamma}")

    