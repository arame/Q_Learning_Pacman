from grid import Pacman_grid
from constants import Constants
from hyper import Hyper

def main():
    p = Pacman_grid()
    for i in range(Hyper.total_episodes):
        p.reset()
        done = False
        while done == False:
            done = p.step()
        episodes = i + 1
        p.print_episode_results(episodes)

if __name__ == "__main__":
    main()