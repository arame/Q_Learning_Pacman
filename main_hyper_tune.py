from grid import Pacman_grid
from config import Hyper, Constants
import optuna
import sys, logging

# This code is the same as in main.py except
# it is using the Optuna library to 
# tune the hyperparameters
# Instead of producing graphs, it produces statistics
def objective(trial):
    Hyper.gamma = trial.suggest_float("Hyper.gamma", 0.01, 0.99)
    Hyper.alpha = trial.suggest_float("Hyper.alpha", 0.01, 0.99)
    Hyper.decay = trial.suggest_float("Hyper.decay", 0.990, 0.999)
    pacman_grid = Pacman_grid()
    for i in range(Hyper.total_episodes):
        pacman_grid.reset()
        done = False
        while done == False:
            if Hyper.is_ghost:
                done = pacman_grid.ghost_step()
            else:
                done = pacman_grid.step()
            pacman_grid.policy.update_epsilon()
        episodes = i + 1
        pacman_grid.print_episode_results(episodes)
        pacman_grid.save_episode_stats()

    # Average the rewards at the end to even out any outlier results caused
    # by the stochastic environment.
    reward_for_last_10_episode = sum(pacman_grid.rewards_per_episode[-10:]) / 10
    return reward_for_last_10_episode
    
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
print("Number of finished trials: ", len(study.trials))
print(study.best_params)
print(study.best_value)     
    