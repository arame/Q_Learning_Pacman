from grid import Pacman_grid
from constants import Constants
from hyper import Hyper

def main():

    p = Pacman_grid()
    for i in range(Hyper.total_episodes):
        no_steps = 0
        p.reset()
        done = False
        while done == False:
            no_steps += 1
            done = p.step()
        episodes = i + 1

        result = "SUCCEEDED"
        print(f"For episode {episodes}, which {result} after {no_steps} steps")

    


if __name__ == "__main__":
    main()