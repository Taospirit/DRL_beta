import gym, os, time
import matplotlib.pyplot as plt
import torch

from AC.ac_net import ActorNet, CriticNet
from AC.ac_policy import A2CPolicy as Policy

env = gym.make('CartPole-v0')
env = env.unwrapped

# env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 32
episodes = 500
max_step = 2000
plot_save_path = './plot/a2c_two_stack'

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticNet(state_space, hidden_dim, 1)
policy = Policy(actor, critic, plot_save_path)


def plot(steps, y_label):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
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
    episodes = 500
    max_step = 2000

    for i in range(1, 6):
        plot_save_path = './plot/a2c_two_stack' + str(i)
        policy = Policy(actor, critic, plot_save_path)

        live_time = []
        for i_eps in range(episodes):
            s = policy.sample(env, max_step)

            live_time.append(s)
            plot(live_time, 'Training_A2C_TwoNet_'+str(max_step))

            policy.learn()
        max_step += 2000

    max_step = 1000000
    plot_save_path = './plot/a2c_two_stack_inf'
    policy = Policy(actor, critic, plot_save_path)

    live_time = []
    for i_eps in range(episodes):
        s = policy.sample(env, max_step)

        live_time.append(s)
        plot(live_time)

        policy.learn()
    
if __name__ == '__main__':
    main()