import gym, os, time
import matplotlib.pyplot as plt
import torch

from AC.ac_net import ActorCriticNet
from AC.ac_policy import OneNetPolicy as Policy

env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 32
episodes = 1000
max_step = 100000
plot_save_path = './plot/ac_one_inf'

ac = ActorCriticNet(state_space, hidden_dim, action_space)
policy = Policy(ac, plot_save_path)


def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training_AC_OneNet')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Step')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_path + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
        # print (f'sava fig in {path}')
    plt.pause(0.0000001)


def main():
    live_time = []
    for i_eps in range(episodes):
        begin = time.time()

        state = env.reset()
        for s in range(max_step):
            action = policy.choose_action(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            
            policy.save_rewards.append(reward)
            state = next_state
            if done:
                break
        live_time.append(s)
        plot(live_time)

        policy.learn()
    
if __name__ == '__main__':
    main()