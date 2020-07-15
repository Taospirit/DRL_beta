import gym
from policy.model_free import A2C
from data.Buffer import Buffer
from data.Batch import Batch
import torch, numpy as np
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        probs = self.actor(input)
        dist = Categorical(probs)
        return dist

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size, std=0.0):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        return self.critic(input)

env = gym.make("CartPole-v0")
num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.n
hidden_size = 256
max_eps = 5000
max_step = 300
buffer_size = 50000
batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer = Buffer(buffer_size)
actor = Actor(num_inputs, num_outputs, hidden_size)
critic = Critic(num_inputs, hidden_size)
policy = A2C(actor, critic, target_update_freq=30)

def check_reward(env, policy, reward, test_eps):
    return False

for eps_num in range(max_eps):

    if check_reward(env, policy, reward, test_eps):
        print ("Finished Learning beyond setting reward")
        break
    
    state = env.reset()
    print (f'Learning eps is {eps_num}')
    for _ in range(max_step):
        
        env.render()

        state = torch.tensor(state).to(device)

        dist = policy.choose_action(state)
        action = dist.sample()
        next_state, reward, done, info = env.step(action.cpu().numpy())
        
        buffer.append(state, action, reward, next_state)

        state = next_state
        if done:
            state = env.reset()

        # if buffer.is_full():
        #     policy.learn(buffer, batch_size, 10)
    # if buffer.is_full():
    policy.learn(buffer, batch_size, 10)

env.close()




        



