import gym, os, time
import matplotlib.pyplot as plt
import torch

from drl.model import ActorDPG, Critic
from drl.policy import DDPGPolicy as Policy
from drl.buffer import ReplayBuffer as Buffer

env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]

hidden_dim = 32
episodes = 1000
max_step = 100
buffer_size = 50000
actor_learn_freq = 1
target_update_freq = 5
batch_size = 300
model_save_dir = './save/test_ddpg'
os.makedirs(model_save_dir, exist_ok=True)

actor = ActorDPG(state_space, hidden_dim, action_space)
critic = Critic(state_space, hidden_dim, action_space)
buffer = Buffer(buffer_size)
policy = Policy(actor, critic, buffer, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq, batch_size=batch_size)

def plot(steps, y_label, plot_save_dir):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Rewards')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
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
    path = dir_path + model_save_dir[1:] + '/' + model_save_dir.split('/')[-1] + '.txt'
    with open(path, 'w+') as f:
        f.write(data)

model = 'eval'
save_file = model_save_dir.split('/')[-1]
def main():
    if model == 'train':
        save_setting()
        live_time = []
        for i_eps in range(episodes):
            step = policy.sample(env, max_step)
            live_time.append(step)
            plot(live_time, 'Training_DDPG_TwoNet_Double', model_save_dir)
            
            policy.learn()
        policy.save_model(model_save_dir, save_file)
    else:
        policy.load_model(model_save_dir, save_file)
        for _ in range(episodes):
            policy.sample(env, max_step, test=True)
        env.close()


if __name__ == '__main__':
    main()