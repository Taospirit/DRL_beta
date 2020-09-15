from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# from drl.model import ActorGaussian, CriticQ
from drl.algorithm import SAC2 as SAC
# from drl.algorithm import SAC


env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2

# assert 0
hidden_dim = 256
episodes = 5000
max_step = 300
buffer_size = 50000
actor_learn_freq = 1
target_update_freq = 10
batch_size = 128

model_save_dir = 'save/test_sac'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ActorGaussian(nn.Module):
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
    
    def action(self, state):
        mean, log_std = self.forward(state)
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

class CriticQTwin(nn.Module):
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

class ValueNet(nn.Module):
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

actor = ActorGaussian(state_space, hidden_dim, action_space)
critic = CriticQTwin(state_space, hidden_dim, action_space)
value_net = ValueNet(state_space)
# buffer = Buffer(buffer_size)
policy = SAC(actor, critic, value_net, action_space=env.action_space, buffer_size=buffer_size,
              actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq, batch_size=batch_size)

def sample(env, policy, max_step, test=False):
    assert env, 'You must set env for sample'
    reward_avg = 0
    state = env.reset()

    for step in range(max_step):
        action = policy.choose_action(state, test)
        # assert  0
        next_state, reward, done, info = env.step(action)
        # env.render()
        # process env callback
        if not test:
            mask = 0 if done else 1
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)

        reward_avg += reward
        if done:
            break
        state = next_state

    if not test:
        policy.learn()
    return reward_avg/(step+1), step


run_type = ['train', 'eval']
run = run_type[0]
plot_name = 'SAC_TwoNet_Double'


def main():
    test = False
    if run == 'eval':
        global episodes
        episodes = 100
        test = True
        print('Loading model...')
        policy.load_model(model_save_dir, save_file, load_actor=True)

    elif run == 'train':
        print('Saving model setting...')
        policy_test.save_setting(env_name, state_space, action_space, episodes,
                                 max_step, policy, model_save_dir, save_file)
    else:
        print('Setting your run type!')
        return 0

    live_time = []
    for i_eps in range(episodes):
        rewards, step = sample(env, policy, max_step, test=test)
        if run == 'eval':
            print(f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)

        if i_eps > 0 and i_eps % 100 == 0:
            print(f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    env.close()


if __name__ == '__main__':
    main()
