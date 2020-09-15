# from test_tool import policy_test
import sys
sys.path.append('/Users/liulintao/drl_repo/DRL_beta/beta')
# sys.path.append('..')
import gym
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

model = namedtuple('model', ['policy_net', 'value_net'])

# from drl.model import ActorNet, CriticV
from drl.algorithm import A2C
from drl.utils import ReplayBuffer as Buffer

class Net(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim)
        )

class ActorNet(Net):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        dist = F.softmax(x, dim=-1)
        return dist

class CriticV(Net):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(output_dim)
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc1 = nn.Linear(22528, 512)
        # self.value_linear = nn.Linear(512, output_dim)

    def forward(self, x):
        # # print (x.size())
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = torch.flatten(x).view(1, -1)
        # # print (f'x.shape {x.size()}')
        # # assert 0
        # x = F.relu(self.fc1(x))
        # state_value = self.value_linear(x)
        # return state_value
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# env
env_name = 'Pong-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
episodes = 2000
max_step = 300
hidden_dim = 32

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, 1)
model = model(actor, critic)
policy = A2C(model, buffer_size=max_step, actor_learn_freq=1, target_update_freq=0)

model_save_dir = 'save/pong'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

writer = SummaryWriter(os.path.dirname(model_save_dir)+'/logs/pong')
train = True

def prepro(obs):
    state = obs.transpose(2, 1, 0)
    # state = np.expand_dims(state, axis=0)
    return state

def main():
    if not train:
        policy.load_model(model_save_dir, save_file, load_actor=True)

    for i_eps in range(episodes):
        obs = env.reset()
        state = prepro(obs)
       
        rewards = 0

        for step in range(max_step):
            action, log_prob = policy.choose_action(np.expand_dims(state, axis=0))
            next_state, reward, done, info = env.step(action)
            if train:
                mask = 0 if done else 1
                policy.process(s=state, r=reward, l=log_prob, m=mask)
            else:
                env.render()
            rewards += reward
            if done:
                break
            state = prepro(next_state)
        if train:
            pg_loss, v_loss = policy.learn()

            writer.add_scalar('reward', rewards, global_step=i_eps)
            writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
            writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)

        if i_eps > 0 and i_eps % 200 == 0:
            print (f'EPS {i_eps}, reward {rewards}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)

    writer.close()
    env.close()

if __name__ == '__main__':
    main()
