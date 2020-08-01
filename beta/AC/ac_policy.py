import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class OneNetPolicy():
    def __init__(self, ac_net, plot_save_path):
        self.lr = 0.01
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.99
        self.save_actions = []
        self.save_rewards = []
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ac_net.to(self.device)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        os.makedirs(plot_save_path, exist_ok=True)

    def choose_action(self, state):
        self.model.eval()

        state = torch.from_numpy(state).float().to(self.device)
        dist, state_value = self.model(state)
        
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)

        self.save_actions.append((log_prob, state_value))

        return action.item()

    def learn(self):
        self.model.train()

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
        # print (f'Type p {type(policy_loss)}, v {type(value_loss)}')
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        # print (f'Type loss {type(loss)}')

        # assert 0
        loss.backward()
        self.optimizer.step()

        self.save_actions = []
        self.save_rewards = []


class TwoNetPolicy():
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
        self.actor.train()
        self.critic.train()

        os.makedirs(plot_save_path, exist_ok=True)

    def choose_action(self, state):
        self.actor.eval()

        # state = torch.from_numpy(state).float().to(self.device)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        # actor
        dist = self.actor(state)
        # critic
        state_value = self.critic(state)
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)

        self.save_actions.append((log_prob, state_value))

        return action.item()

    def learn(self):
        self.actor.train()
        self.critic.train()

        R = 0
        policy_loss = []
        value_loss = []
        rewards = []

        for r in self.save_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        # normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for (log_prob, value), r in zip(self.save_actions, rewards):
            td_error = r - value.item() # td_error
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

    def sample(self, env, max_steps):
        assert env, 'You must set env for sample'

        state = env.reset()
        for i in range(max_steps):

            action = self.choose_action(state)

            next_state, reward, done, info = env.step(action)
            env.render()

            self.process(reward)
            
            state = next_state
            if done:
                # state = env.reset()
                break
        return i

    def process(self, r):
        self.save_rewards.append(r)


class DoublePolicy():
    def __init__(self, actor_net, critic_net, target_update_fre=0):
        assert target_update_fre > 0, "you must set target_update_fre not zero"

    def choose_action(self):
        pass

    def learn(self):
        pass

class A2CPolicy(): #trick: double, soft_update
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        plot_save_path,
        target_update_fre=0,
        learning_rate = 0.01
        discount_factor = 0.99
        ):
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = discount_factor

        self.save_log_probs = []
        self.save_values = []
        self.save_rewards = []
        self.save_masks = []
        self.next_state = None
        # self.end_value = 0
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device)
        self.critic_eval = critic_net.to(self.device)
        self.actor_eval_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.critic_target = deepcopy(self.critic_eval)
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

            self.actor_target.eval()
            self.critic_target.eval()
            
        # 什么时候用eval?什么适合用train?
        
        #TODO
        self.criterion = nn.SmoothL1Loss()

        os.makedirs(plot_save_path, exist_ok=True)

    def choose_action(self, state, test=False):
        if test:
            self.actor_eval.eval()

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = self.actor_eval(state)
        m = Categorical(dist)
        action = m.sample()
        if test:
            return action.item()

        log_prob = m.log_prob(action)
        # critic
        state_value = self.critic_eval(state)

        self.save_log_probs.append(log_prob)
        self.save_values.append(state_value)

        return action.item()

    def learn(self):
        critic = self.critic_target if self._target else self.critic_eval
        next_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device)
        
        end_value = critic(next_state).item()

        
        def compute_returns(next_value, rewards, masks, gamma=0.99):
            assert len(rewards) == len(masks), 'rewards & masks must have same length'
            R = next_value
            returns = []
            for step in reversed(range(len(rewards))):
                R = rewards[step] + gamma * R * masks[step]
                returns.insert(0, R)
            return returns
        
        self.save_rewards = compute_returns(end_value, self.save_rewards, self.save_masks, self.gamma)
        assert len(self.save_rewards) == len(self.save_values) == len(self.save_log_probs)

        # policy_loss, value_loss = [], []
        # for r, v, log in zip(returns, self.save_values, self.save_log_probs):
        #     advantage = r - v.detach()
        #     policy_loss.append(-log * advantage)
        #     value_loss.append(F.smooth_l1_loss(r, v))
        # policy_loss = torch.stack(policy_loss)
        # value_loss  = torch.stack(value_loss)

        log_probs = torch.stack(self.save_log_probs).unsqueeze(0) # [1, len(...)]
        values_target = torch.stack(self.save_rewards).unsqueeze(0).to(self.device)
        values = torch.stack(self.save_values).reshape(1, -1)
        
        advantage = values_target - values.detach()
        policy_loss = -log_probs * advantage

        value_loss = torch.stack([F.smooth_l1_loss(r, v) for r, v in zip(self.save_rewards, self.save_values)])

        self.actor_eval.train()
        self.critic.train()

        self.actor_eval_optim.zero_grad()
        actor_loss = policy_loss.sum()
        actor_loss.backward()
        self.actor_eval_optim.step()

        self.critic_eval_optim.zero_grad()
        critic_loss = value_loss.sum()
        critic_loss.backward()
        self.critic_eval_optim.step()

        self.save_log_probs = []
        self.save_masks = []
        self.save_rewards = []
        self.save_values = []

    def sample(self, env, max_steps):
        assert env, 'You must set env for sample'

        state = env.reset()
        for i in range(max_steps):

            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            # process env callback
            self.process(reward, done)
            state = next_state

            if done:
                state = env.reset()
                break

        self.next_state = state
        return i


    def process(self, reward, done):
        # print (type(reward), reward)
        done = 0 if done else 1
        # put var in cpu
        reward = torch.tensor(reward, dtype=torch.float32)
        mask = torch.tensor(done, dtype=torch.float32)
        # print (f'Device r {reward.device}, m {mask.device}')
        self.save_rewards.append(reward)
        self.save_masks.append(mask)