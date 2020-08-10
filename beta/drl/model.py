import gym, os, time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        dist = F.softmax(action_score, dim=-1)
        return dist, state_value

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        dist = F.softmax(action_score, dim=-1)
        return dist

# DDPG
class ActorDPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
    
    def forward(self, state, noise_std=0.0):
        action = self.net(state)
        return action

# PPO
class ActorPPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.mu_head(x)) # test for gym_env: 'Pendulum-v0'
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma
 
class CriticV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value_head(x)
        return state_value

class CriticQ(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # inpur_dim = state_dim + action_dim, 
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        q_value = self.net(input)
        return q_value
