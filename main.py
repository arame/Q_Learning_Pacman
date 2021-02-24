from grid import Pacman_grid
from constants import Constants
from hyper import Hyper

def main():
    print("\n"*10)
    print("-"*100)
    print("Start of QLearning Basic design for Pacman")
    Hyper.display()
    print("-"*100)
    p = Pacman_grid()
    for i in range(Hyper.total_episodes):
        p.reset()
        done = False
        while done == False:
            done = p.step()
        episodes = i + 1
        p.print_episode_results(episodes)

    print("\n"*5)  
    print("-"*100)
    Hyper.display()
    print("End of QLearning Basic design for Pacman")
    print("-"*100)
if __name__ == "__main__":
    main()