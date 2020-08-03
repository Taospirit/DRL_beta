import gym, os, time
import matplotlib.pyplot as plt
import torch

from drl.model import ActorNet, CriticNet
from drl.policy import A2CPolicy as Policy

env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 32
episodes = 300
max_step = 1000
# default
actor_learn_freq=1
target_update_freq=0

plot_save_dir = './save/a2c_test'
os.makedirs(plot_save_dir, exist_ok=True)

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticNet(state_space, hidden_dim, 1)
policy = Policy(actor, critic, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq)


def plot(steps, y_label, plot_save_dir):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Step')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 50 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)

def save_setting():
    line = '===============================\n'
    env_info = f'env: {env_name} \nstate_space: {state_space}, action_space: {action_space}\n' 
    env_info += f'episodes: {episodes}, max_step: {max_step}\n'
    policy_dict = vars(policy)
    policy_info = ''
    for item in policy_dict.keys():
        policy_info += f'{item}: {policy_dict[item]} \n'

    data = line.join([env_info, policy_info])

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = dir_path + plot_save_dir[1:] + '/' + plot_save_dir.split('/')[-1] + '.txt'
    with open(path, 'w+') as f:
        f.write(data)

def main():
    save_setting()
    live_time = []
    for i_eps in range(episodes):
        step = policy.sample(env, max_step)
        live_time.append(step)
        plot(live_time, 'Training_A2C_TwoNet_no_Double', plot_save_dir)
        
        policy.learn()
    policy.save_model(plot_save_dir, plot_save_dir.split('/')[-1])

if __name__ == '__main__':
    main()