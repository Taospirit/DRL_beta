import gym
from drl.policy import A2C
from drl.data import Batch, Buffer
# from drl.data.batch import Batch
# from drl.data.buffer import Buffer
import torch, numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import os
model_path = os.path.dirname(__file__) + '/test_a2c.pth'


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        probs = self.actor(x)
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

    def forward(self, x):
        return self.critic(x)

env = gym.make("CartPole-v0")
env.seed(1)
torch.manual_seed(1)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
hidden_size = 256
max_eps = 10000
max_step = 10
buffer_size = 10000
batch_size = 20

buffer = Buffer(buffer_size)
actor = Actor(num_inputs, num_outputs, hidden_size)
critic = Critic(num_inputs, hidden_size)
policy = A2C(actor, critic, actor_learn_freq=5, target_update_freq=10)

def check_reward(env, policy, test_eps_num, threshold):
    r_list = []
    r_sum, eps_num = 0, 0
    state = env.reset()
    while eps_num < test_eps_num:
        print (f'test eps {eps_num}', end='\r')
        env.render()

        action = policy.choose_action(state)
        next_state, reward, done, info = env.step(action)
        r_sum += reward

        if done:
            r_list.append(r_sum)
            # print (f"done in {_}, eps rewards is {r_sum}")
            r_sum = 0
            eps_num += 1
            state = env.reset()

        state = next_state

    train_done = np.mean(r_list) > threshold
    reward_mean = np.mean(r_list)
    return reward_mean, train_done

# train_type = 'eval'

def train_model():
    for eps_num in range(max_eps):
        
        reward_avg, train_done = check_reward(env, policy, test_eps_num=3, threshold=200)

        if train_done:
            print (f"*******Finished Learning for reward {reward_avg}********")
            break
        
        state = env.reset()
        print (f'---------Learning eps is {eps_num+1}, reward_avg is {reward_avg}---------')
        for _ in range(max_step):
            print (f'training step: {_}', end='\r')
            env.render()

            # state = torch.tensor(state).to(device)

            action = policy.choose_action(state)
            # action = dist.sample()
            next_state, reward, done, info = env.step(action)
            
            buffer.append(state, action, reward, next_state)

            state = next_state
            if done:
                state = env.reset()

            # if buffer.is_full():
            #     policy.learn(buffer, batch_size, 10)
        # if buffer.is_full():
        policy.learn(buffer, batch_size)

    print ('Learing Over!')
    policy.save_model(model_path)

    env.close()

def show_model():
    policy.load_model(model_path)
    print (f'-----Load model from {model_path}')
    state = env.reset()
    r_sum = 0
    for _ in range(10000):
        # print ()
        env.render()
        # state = env.reset()
        action = policy.choose_action(state)
        next_state, reward, done, info = env.step(action)
        r_sum += reward
        # r_sum.append(reward)
        if done:
            print (f"done in {_}, eps rewards is {r_sum}")
            r_sum = 0
            state = env.reset()
        state = next_state

# if __name__=='main':
# show_model()
train_model()


