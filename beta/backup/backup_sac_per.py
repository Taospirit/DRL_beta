# from test_tool import policy_test
import gym
import os
from os.path import abspath, dirname
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
from drl.algorithm import SACV as SAC
from utils.plot import plot

env_name = 'Pendulum-v0'
buffer_size = 50000
actor_learn_freq = 1
update_iteration = 10
target_update_freq = 10
batch_size = 128
hidden_dim = 32
episodes = 2000 + 10
max_step = 300
lr = 1e-3

LOG_DIR = '/logs'
SAVE_DIR = '/save/sac_per_'
POLT_NAME = 'SAC_per'
PKL_DIR = '/pkl/sac_per'

use_priority = True
if use_priority:
    p = 'per_'
    SAVE_DIR += p
    POLT_NAME += p
    PKL_DIR += p

POLT_NAME += env_name
SAVE_DIR += env_name
PKL_DIR += env_name

file_path = abspath(dirname(__file__))
pkl_dir = file_path + PKL_DIR
model_save_dir = file_path + SAVE_DIR
save_file = model_save_dir.split('/')[-1]
writer_path = model_save_dir + LOG_DIR

try:
    os.makedirs(model_save_dir)
except FileExistsError:
    import shutil
    shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)

env = gym.make(env_name)
env = env.unwrapped

# Parameters
state_space = env.observation_space.shape[0]
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
        self.net1 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        self.net2 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q1_value = self.net1(x)
        q2_value = self.net2(x)
        return q1_value, q2_value


class CriticModelDist(nn.Module):
    def __init__(self, obs_dim, mid_dim, act_dim, v_min, v_max, num_atoms=51):
        self.net1 = nn.Sequential(nn.Linear(obs_dim + act_dim , mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, num_atoms), )
        self.net2 = nn.Sequential(nn.Linear(obs_dim + act_dim , mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, num_atoms), )

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        # self.fc1 = nn.Linear(obs_dim + act_dim, mid_dim)
        # self.fc2 = nn.Linear(mid_dim, mid_dim)
        # self.fc3 = nn.Linear(mid_dim, num_atoms)
        
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        z1 = self.net1(x)
        z2 = self.net2(x)
        return z1, z2

    def get_probs(self, obs, act):
        z1, z2 = self.forward(obs, act)
        z1 = torch.log_softmax(z1, dim=1)
        z2 = torch.log_softmax(z2, dim=1)
        return z1, z2

class ValueModel(nn.Module):
    def __init__(self, state_dim, init_w=3e-3):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        layer_norm(self.linear1, std=1.0)
        layer_norm(self.linear2, std=1.0)
        layer_norm(self.linear3, std=1.0)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

TRAIN = True
PLOT = True
WRITER = False

def sample(env, policy, max_step):
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
        if done:
            break
        state = next_state
    return rewards

def train():
    mean, std = [], []
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    live_time = []

    # while policy.warm_up():
    #     sample(env, policy, max_step)
    #     print (f'Warm up for buffer {policy.buffer.size()}', end='\r')

    for i_eps in range(episodes):
        rewards = sample(env, policy, max_step)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        mean.append(reward_mean)
        std.append(reward_std)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{round(reward_mean, 3)}')
        else:
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
    env.close()
    return mean, std

if __name__ == '__main__':
    model = namedtuple('model', ['policy_net', 'value_net', 'v_net'])
    actor = ActorModel(state_space, hidden_dim, action_space)
    critic = CriticModel(state_space, hidden_dim, action_space)
    v_net = ValueModel(state_space)
    rl_agent = model(actor, critic, v_net)
    policy = SAC(rl_agent, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq,
            update_iteration=update_iteration, target_update_freq=target_update_freq, 
            batch_size=size, use_priority=use_priority)
    writer = SummaryWriter(writer_path)
    train()