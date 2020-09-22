from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import numpy as np
# from drl.model import ActorModel, CriticQ
# from drl.algorithm import SAC
from drl.algorithm import SAC_PER as SAC
from utils.plot import plot

env_list = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'MountainCar-v0']
env_name = env_list[0]
env = gym.make(env_name)
env = env.unwrapped


# Parameters
state_space = env.observation_space.shape[0]
print (env.action_space)
action_space = env.action_space.shape[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ActorModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer_norm(self.fc1, std=1.0)
        layer_norm(self.fc2, std=1.0)
        layer_norm(self.mean, std=1.0)
        layer_norm(self.log_std, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def action(self, state, test=False):
        mean, log_std = self.forward(state)
        if test:
            return mean.detach().cpu().numpy()
        std = log_std.exp()
        normal = Normal(mean, std)
        
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        action = action * action_scale + action_bias
        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0,1)
        
        z = noise.sample()
        action = torch.tanh(mean + std*z.to(self.device))
        log_prob = normal.log_prob(mean + std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        
        return action, log_prob

class CriticModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # inpur_dim = state_dim + action_dim, 
        self.net1 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        self.net2 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        # self.net = build_critic_network(state_dim, hidden_dim, action_dim)

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        q1_value = self.net1(input)
        q2_value = self.net2(input)
        return q1_value, q2_value

class ValueModel(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        layer_norm(self.linear1, std=1.0)
        layer_norm(self.linear2, std=1.0)
        layer_norm(self.linear3, std=1.0)

        # self.linear3.weight.data.uniform_(-edge, edge)
        # self.linear3.bias.data.uniform_(-edge, edge)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

file_path = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = '/save/test_sac2_'
LOG_DIR = '/runs'
POLT_NAME = 'SAC_'
PKL_NAME = '/sac.pkl'

use_priority = False
if use_priority:
    SAVE_DIR += 'per_'
    POLT_NAME += 'per_'
    PKL_NAME = '/sac_per.pkl'

POLT_NAME += env_name
SAVE_DIR += env_name

model_save_dir = file_path + SAVE_DIR
save_file = model_save_dir.split('/')[-1]

try:
    os.makedirs(model_save_dir)
except FileExistsError:
    import shutil
    shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)

hidden_dim = 256
episodes = 500
max_step = 300

writer_path = model_save_dir + LOG_DIR
writer = SummaryWriter(writer_path)

model = namedtuple('model', ['policy_net', 'value_net', 'v_net'])

TRAIN = True
PLOT = True
WRITER = False

def sample(env, policy, max_step):
    # rewards = 0
    rewards = []
    state = env.reset()
    for step in range(max_step):
        #==============choose_action==============
        action = policy.choose_action(state)
        next_state, reward, done, info = env.step(action)
        if TRAIN:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
        else:
            env.render()
        rewards.append(reward)
        # rewards += reward
        if done:
            break
        state = next_state

    return rewards

def train():
    mean, std = [], []
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    live_time = []
    
    while policy.warm_up():
        sample(env, policy, max_step)
        print (f'Warm up for buffer {policy.buffer.size()}', end='\r')

    for i_eps in range(episodes+10):
        rewards = sample(env, policy, max_step)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        mean.append(reward_mean)
        std.append(reward_std)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{round(reward_mean, 3)}')
        else:
            # if policy.warm_up():
            #     i_eps = 0
            #     print (f'Warm up for buffer {policy.buffer.size()}', end='\r')
            #     continue
            #==============learn==============
            pg_loss, q_loss, a_loss = policy.learn()
            if PLOT:
                live_time.append(reward_mean)
                plot(live_time, POLT_NAME, model_save_dir, 100)
            if WRITER:
                writer.add_scalar('reward', reward_mean, global_step=i_eps)
                writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
                writer.add_scalar('loss/q_loss', q_loss, global_step=i_eps)
                writer.add_scalar('loss/alpha_loss', a_loss, global_step=i_eps)

            if i_eps % 5 == 0:
                print (f'EPS:{i_eps}, reward_mean:{round(reward_mean, 3)}, pg_loss:{round(pg_loss, 3)}, q_loss:{round(q_loss, 3)}, alpha_loss:{round(a_loss, 3)}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    writer.close()
    env.close
    return mean, std

if __name__ == '__main__':
    means, stds = [], []

    for seed in range(5):
        env.seed(seed  * 10)
        torch.manual_seed(seed * 10)

        actor = ActorModel(state_space, hidden_dim, action_space)
        critic = CriticModel(state_space, hidden_dim, action_space)
        v_net = ValueModel(state_space)
        rl_agent = model(actor, critic, v_net)
        policy = SAC(rl_agent, buffer_size=50000, actor_learn_freq=5, update_iteration=10,
                target_update_freq=20, batch_size=128, use_priority=use_priority)

        mean, std = train()
        means.append(mean)
        stds.append(std)

    d = {'mean': means, 'std': stds}
    pkl_dir = file_path + PKL_NAME
    print (pkl_dir)
    import pickle
    with open(pkl_dir, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)