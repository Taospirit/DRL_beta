import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        action = F.softmax(action_score, dim=-1)
        return action

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value_head(x)
        return state_value

class Policy():
    def __init__(self, actor_net, critic_net, plot_save_path):
        self.lr = 0.01
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.99
        self.save_actions = []
        self.save_rewards = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor_net.to(self.device)
        self.critic = critic_net.to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        os.makedirs(plot_save_path, exist_ok=True)


    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        # actor
        probs = self.actor(state)
        # critic
        state_value = self.critic(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        self.save_actions.append((log_prob, state_value))

        return action.item()

    def learn(self):
        R = 0
        policy_loss = []
        value_loss = []
        rewards = []

        for r in self.save_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        # advantage?
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for (log_prob, value), r in zip(self.save_actions, rewards):
            td_error = r - value.item()
            policy_loss.append(-log_prob * td_error)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r]).to(self.device)))

        # print (f'critic_loss is {np.sum(value_loss)}, actor_loss is {np.sum(policy_loss)}')

        self.actor_optim.zero_grad()
        actor_loss = torch.stack(policy_loss).sum()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss = torch.stack(value_loss).sum()
        critic_loss.backward()
        self.critic_optim.step()

        self.save_actions = []
        self.save_rewards = []


env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 32
plot_save_path = './AC_two/'
episodes = 5000
max_step = 3000

actor = Actor(state_space, hidden_dim, action_space)
critic = Critic(state_space, hidden_dim, 1)
policy = Policy(actor, critic, plot_save_path)


def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training_AC_TwoNet')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Step')
    ax.plot(steps)
    RunTime = len(steps)

    path = plot_save_path + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
        print (f'sava fig in {path}')
    plt.pause(0.0000001)


def main():
    live_time = []
    for i_eps in range(episodes):
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