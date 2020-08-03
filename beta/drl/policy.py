import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, **kwargs):
        super().__init__()
    
    @abstractmethod
    def choose_action(self, state, **kwargs):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def process(self, **kwargs):
        pass
    
    @abstractmethod
    def sample(self, env, max_steps, **kwargs):
        pass

    def save_model(self, save_dir, save_file_name):
        assert isinstance(save_dir, str) and isinstance(save_file_name, str)
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + '/' + save_file_name + '.pth'
        torch.save(self.actor_eval.state_dict(), save_path)

    def load_model(self, path):
        assert isinstance(path, str)
        assert path.split('.')[-1] == 'pth', "Why not a .pth file? Are U TFbooys?"
        self.actor_eval.load_state_dict(torch.load(path))

class A2CPolicy(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        actor_learn_freq=1,
        target_update_freq=0,
        soft_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = soft_tau
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

    @staticmethod
    def soft_sync_weight(target, source, tau=0.01):
        with torch.no_grad():
            for target_param, eval_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    def sample(self, env, max_steps):
        assert env, 'You must set env for sample'
        state = env.reset()
        for i in range(max_steps):
            action = self.choose_action(state)

            next_state, reward, done, info = env.step(action)
            env.render()
            # process env callback
            self.process(s=state, a=action, s_=next_state, r=reward, d=done, i=info)

            if done:
                state = env.reset()
                break
            state = next_state
        self.next_state = state
        
        return i

    def process(self, **kwargs):
        reward, done = kwargs['r'], kwargs['d']

        mask = 0 if done else 1
        reward = torch.tensor(reward, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        self.save_eps['rewards'].append(reward)
        self.save_eps['masks'].append(mask)



class DDPGPolicy(BasePolicy):
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        buffer,
        actor_learn_freq=1,
        target_update_freq=0,
        learning_rate=0.01,
        discount_factor=0.99,
        batch_size=100,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()

        self.next_state = None
        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._update_iteration = 10
        self._sync_cnt = 0
        self._learn_cnt = 0
        # self._learn_critic_cnt = 0
        # self._learn_actor_cnt = 0
        self._verbose = verbose
        self._buffer = buffer
        self._batch_size = batch_size
        assert not buffer, 'You must set Buffer to DDPG'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device) # pi(s)
        self.critic_eval = critic_net.to(self.device) # Q(s, a)
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
            
        self.criterion = nn.MSELoss() # why mse?

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if test:
            self.actor_eval.eval()

        action = self.actor_eval(state) # out = tanh(x)
        return action.cpu().data.numpy()
        # return action.cpu().data.numpy().flatten()

    def learn(self):
        loss_actor_avg, loss_critic_avg = 0, 0

        for _ in range(self._update_iteration):
            memory_batchs = self._buffer.random_sample(self._batch_size)
            state, action, next_state, reward = self._buffer.split(memory_batchs)

            S = torch.tensor(state, dtype=torch.float32, device=self.device) # [batch_size, S.feature_size]
            A = torch.tensor(action, dtype=torch.float32, device=self.device)
            S_ = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            # d = torch.tensor(done, dtype=torch.float32, device=self.device) # ?
            R = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
            # print (f'Size S_ {S_.size()}')
            with torch.no_grad():
                q_target = self.critic_eval(S_, self.actor_eval(S_))
                if self._target:
                    q_target = self.critic_target(S_, self.actor_target(S_))
                # q_target = r + ((1 - d) * self._gamma * q_target) # (1 - d)
                q_target = R + self._gamma * q_target
                # print (f'Size R {R.size()}, q_target {q_target.size()}')

            q_eval = self.critic_eval(S, A) # [batch_size, q_value_size]
            critic_loss = self.criterion(q_eval, q_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_cnt += 1

            if self._learn_cnt % self.actor_learn_freq == 0:
                actor_loss = -self.critic_eval(S, self.actor_eval(S)).mean()
                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()

            if self._target:
                if self._learn_cnt % self.target_update_freq == 0:
                    if self._verbose: print (f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval)
                    self.soft_sync_weight(self.actor_target, self.actor_eval)
        
        loss_actor_avg /= self._update_iteration
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg

    @staticmethod
    def soft_sync_weight(target, source, tau=0.01):
        with torch.no_grad():
            for target_param, eval_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    def sample(self, env, max_steps):
        assert env, 'You must set env for sample'
        rewards = 0
        state = env.reset()
        for i in range(max_steps):
            action = self.choose_action(state)
            action = action.clip(-1, 1)
            action_max = env.action_space.high[0]

            next_state, reward, done, info = env.step(action * action_max)
            env.render()
            self.process(s=state, a=action, s_=next_state, r=reward, d=done, i=info)
            rewards += reward
            # process env callback
            if done:
                state = env.reset()
                break
            state = next_state

        return rewards

    def process(self, **kwargs):
        state, action, next_state, reward = kwargs['s'], kwargs['a'], kwargs['s_'], kwargs['r']
        self._buffer.append(state, action, next_state, reward)