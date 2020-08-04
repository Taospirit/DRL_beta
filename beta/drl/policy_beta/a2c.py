import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from base import BasePolicy

class A2CPolicy(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.save_eps = {'log_probs':[], 'values':[], 'rewards':[], 'masks': []}

        self.next_state = None
        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._sync_cnt = 0
        self._learn_cnt = 0
        # self._learn_critic_cnt = 0
        # self._learn_actor_cnt = 0
        self._verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device)
        self.critic_eval = critic_net.to(self.device)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.critic_target = deepcopy(self.critic_eval)
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

            self.actor_target.eval()
            self.critic_target.eval()

        self.criterion = nn.SmoothL1Loss()

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if test:
            self.actor_eval.eval()
            return Categorical(self.actor_eval(state)).sample().item()
        
        dist = self.actor_eval(state)
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)
        state_value = self.critic_eval(state)

        self.save_eps['log_probs'].append(log_prob)
        self.save_eps['values'].append(state_value)

        return action.item()

    def learn(self):
        next_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device)
        critic = self.critic_target if self._target else self.critic_eval
        
        with torch.no_grad():
            next_value = critic(next_state)
            def compute_returns(next_value, rewards, masks, gamma=0.99):
                assert len(rewards) == len(masks), 'rewards & masks must have same length'
                R = next_value
                returns = []
                for step in reversed(range(len(rewards))):
                    R = rewards[step] + gamma * R * masks[step]
                    returns.insert(0, R)
                return returns
            v_target = compute_returns(next_value, self.save_eps['rewards'], self.save_eps['masks'])
        
        assert len(self.save_eps['values']) == len(self.save_eps['rewards']), "Error: not same size"
   
        v_eval = torch.stack(self.save_eps['values']).reshape(1, -1) # values = torch.stack(self.save_values, dim=1)
        v_target = torch.stack(v_target).reshape(1, -1).to(self.device)
        critic_loss = self.criterion(v_eval, v_target)

        self.critic_eval.train()
        self.critic_eval_optim.zero_grad()
        critic_loss.backward()
        self.critic_eval_optim.step()
        self._learn_cnt += 1

        if self._learn_cnt % self.actor_learn_freq == 0:
            log_probs = torch.stack(self.save_eps['log_probs']).unsqueeze(0) # [1, len(...)]
            advantage = v_target - v_eval.detach()
            actor_loss = (-log_probs * advantage).sum()
            # print (f'Size log {log_probs.size()}, value_ {v_target.size()}, value {v_eval.size()}, advantage {advantage.size()}, actor_loss {actor_loss.size()}')
            self.actor_eval.train()
            self.actor_eval_optim.zero_grad()
            actor_loss.backward()
            self.actor_eval_optim.step()

        if self._target:
            if self._learn_cnt % self.target_update_freq == 0:
                if self._verbose: print (f'=======Soft_sync_weight of AC=======')
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        
        self.save_eps = {'log_probs':[], 'values':[], 'rewards':[], 'masks': []}

    def sample(self, env, max_steps, test=False):
        assert env, 'You must set env for sample'
        reward_avg = 0
        state = env.reset()
        for step in range(max_steps):
            action = self.choose_action(state, test)

            next_state, reward, done, info = env.step(action)
            env.render()
            # process env callback
            self.process(s=state, a=action, s_=next_state, r=reward, d=done, i=info)
            reward_avg += reward

            if done:
                state = env.reset()
                break
            state = next_state
        self.next_state = state
        if self._verbose: print (f'------End eps at {step} steps------')

        return reward_avg/step

    def process(self, **kwargs):
        reward, done = kwargs['r'], kwargs['d']

        mask = 0 if done else 1
        reward = torch.tensor(reward, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        self.save_eps['rewards'].append(reward)
        self.save_eps['masks'].append(mask)
