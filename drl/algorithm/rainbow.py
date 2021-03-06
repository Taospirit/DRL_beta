import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

#TODO:to be complete
class Rainbow(BasePolicy): #option: double(done), dueling(todo), noisy(todo), n-step(todo), 
    def __init__(
        self, 
        critic_net,
        action_shape=0,
        buffer_size=1000,
        batch_size=100,
        target_update_freq=1,
        target_update_tau=1,
        learning_rate=0.01,
        discount_factor=0.99,
        verbose=False
        ):
        super().__init__()
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.epsilon = 0.5
        # ticks
        self.double_q = True
        self.dueling_q = True
        self.distributional_q = True
        self.prioritized_replay = True
        self.noisy_q = True
        self.n_step_td = True

        self.target_update_freq = target_update_freq
        self.action_shape = action_shape
        self._gamma = discount_factor
        self._batch_size = batch_size
        self._verbose=verbose
        self._update_iteration = 10
        self._learn_cnt = 0
        self._normalized=lambda x, e: (x - x.mean()) / (x.std() + e)
        self.rew_norm = True
        self.buffer = ReplayBuffer(buffer_size)
        
        self.critic_eval = critic_net.to(self.device)
        self.critic_target = deepcopy(self.critic_eval)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_eval.use_dueling = self.critic_target.use_dueling = self.dueling_q # Dueling DQN  

        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.critic_eval.train()
        
        self.criterion = nn.MSELoss()

        self.random_choose = 0
        self.sum_choose = 0

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.critic_eval(state)
        action = q_values.argmax(dim=1).cpu().data.numpy()
        action = action[0] if self.action_shape == 0 else action.reshape(self.action_shape)  # return the argmax index

        if test: self.epsilon = 1.0
        if np.random.randn() >= self.epsilon: # epsilon-greedy
            self.random_choose += 1
            action = np.random.randint(0, q_values.size()[-1])
            action = action if self.action_shape == 0 else action.reshape(self.action_shape)

        self.sum_choose += 1
        return action

    def learn(self):
        for _ in range(self._update_iteration):
            batch_split = self.buffer.split_batch(self._batch_size) # s, a, r, s_
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device)
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
            M = torch.tensor(batch_split['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch_split['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)
            # print (f'SIZE S {S.size()}, A {A.size()}, M {M.size()}, R {R.size()}, S_ {S_.size()}')
            if self.rew_norm: R = self._normalized(R, self.eps)

            with torch.no_grad():
                get_action_net = self.critic_eval if self.double_q else self.critic_target # Double DQN
                argmax_action = get_action_net(S_).max(dim=1, keepdim=True)[1]
                q_next = self.critic_target(S_).gather(1, argmax_action.type(torch.long))
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(self.device)
                
            q_eval = self.critic_eval(S).gather(1, A.type(torch.long))

            critic_loss = self.criterion(q_eval, q_target)
            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()

            self._learn_cnt += 1
            if self._learn_cnt % self.target_update_freq == 0:
                if self._verbose: print (f'=======Soft_sync_weight of DQN=======')
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                # self.epsilon += 0.01
                # self.epsilon = min(self.epsilon, 1.0)
        # print (f'Random {self.random_choose}, Sum {self.sum_choose}, ratio {self.random_choose/self.sum_choose}, epsilon {self.epsilon}')