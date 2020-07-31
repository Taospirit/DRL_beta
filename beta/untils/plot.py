import os
import matplotlib.pyplot as plt

class plot():
    def __init__(self, nums):
        ax = plt.subplot(111)
        ax.cla()
        ax.grid()
        ax.set_title('Training_AC_OneNet')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Run Step')
        ax.plot(steps)
        RunTime = len(steps)

        path = plot_save_path + 'RunTime' + str(RunTime) + '.jpg'
        if len(steps) % 100 == 0:
            plt.savefig(path)
            # print (f'sava fig in {path}')
        plt.pause(0.0000001)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training_AC_OneNet')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Step')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_path + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
        # print (f'sava fig in {path}')
    plt.pause(0.0000001)
