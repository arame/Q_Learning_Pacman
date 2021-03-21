from grid import Pacman_grid
from config import Hyper, Constants

# It all starts here
def main():
    print("\n"*10)
    print("-"*100)
    print("Start of QLearning Basic design for Pacman")
    Hyper.display()
    print("-"*100)
    pacman_grid = Pacman_grid()
    for i in range(Hyper.total_episodes):
        pacman_grid.reset()
        done = False
        while done == False:
            if Hyper.is_ghost:
                done = pacman_grid.ghost_step(i)
            else:
                done = pacman_grid.step(i)
            pacman_grid.policy.update_epsilon()
        episodes = i + 1
        pacman_grid.print_episode_results(episodes)
        pacman_grid.save_episode_stats()

    pacman_grid.print_results()
    print("\n"*5)  
    print("-"*100)
    Hyper.display()
    print("End of QLearning Basic design for Pacman")
    print("-"*100)
    
if __name__ == "__main__":
    main()