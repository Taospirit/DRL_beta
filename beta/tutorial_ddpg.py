from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple

from drl.algorithm import DDPG
from utils.plot import plot
# env
env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]

class ActorDPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        action = self.net(state)
        return action

    def action(self, state, noise_std=0, noise_clip=0.5):
        action = self.net(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(self.device)
            action += noise_norm.clamp(-noise_clip, noise_clip)
        action *= action_max
        action = action.clamp(-action_max, action_max)
        return action.item()

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


hidden_dim = 32
episodes = 5000
max_step = 200

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorDPG(state_space, hidden_dim, action_space)
critic = CriticQ(state_space, hidden_dim, action_space)
model = model(actor, critic)
policy = DDPG(model, buffer_size=50000, actor_learn_freq=1, target_update_freq=10, batch_size=1000)

model_save_dir = 'save/ddpg'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

writer = SummaryWriter(os.path.dirname(model_save_dir)+'/logs/ddpg_1')

TRAIN = True
PLOT = True
WRITER = False

def sample(env, policy, max_step):
    reward_avg = 0
    state = env.reset()
    for step in range(max_step):
        #==============choose_action==============
        action = policy.choose_action(state)
        next_state, reward, done, info = env.step([action])
        if TRAIN:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
        else:
            env.render()
        reward_avg += reward
        if done:
            break
        state = next_state
    reward_avg /= (step + 1)
    return reward_avg

def main():
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    live_time = []
    for i_eps in range(episodes):
        reward_avg = sample(env, policy, max_step)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{round(reward_avg, 3)}')
        else:
            #==============learn==============
            pg_loss, v_loss = policy.learn()
            if PLOT:
                live_time.append(reward_avg)
                plot(live_time, 'DDPG_'+env_name, model_save_dir)
            if WRITER:
                writer.add_scalar('reward', reward_avg, global_step=i_eps)
                writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
                writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
            if i_eps % 5 == 0:
                print (f'EPS:{i_eps}, reward_avg:{round(reward_avg, 3)}, pg_loss:{round(pg_loss, 3)}, v_loss:{round(v_loss, 3)}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    writer.close()
    env.close()

if __name__ == '__main__':
    main()
