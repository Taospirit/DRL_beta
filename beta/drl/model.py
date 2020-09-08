import gym, os, time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

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

class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_mean = nn.Linear(512, 2)
        self.fc4_logstd = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=-20, max=2)
        return a_mean, a_logstd
    
    def sample(self, s):
        a_mean, a_logstd = self.forward(s)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)

# DDPG & TD3
class ActorDPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.net(state)
        return action

    def predict(self, state, action_max, noise_std=0, noise_clip=0.5):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.net(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(self.device)
            action += noise_norm.clamp(-noise_clip, noise_clip)

        action = action.clamp(-action_max, action_max)
        return action

# PPO
class ActorPPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.mu_head, std=1.0)
            self.layer_norm(self.sigma_head, std=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        # x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.mu_head(x)) # test for gym_env: 'Pendulum-v0'
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma
    
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

class CriticV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, output_dim)
        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.value_head, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value_head(x)
        return state_value

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

class CriticDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, atoms=51, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, action_dim)
        self.v_value = nn.Linear(hidden_dim, 1)
        self.use_dueling = False
        self.use_distributional = False

        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.q_value, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.use_dueling:
            v_value = self.v_value(x)
            adv = self.q_value(x)
            return v_value + adv - adv.mean()
        
        if self.use_distributional:
            self.q_value = nn.Linear(hidden_dim, action_dim*atoms)
            x = self.q_value(x)
            return F.softmax(x.view(-1, action_dim, atoms), dim=2)

        q_value = self.q_value(x)
        return q_value

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

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
