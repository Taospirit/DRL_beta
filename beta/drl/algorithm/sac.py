import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

class SAC(BasePolicy):
    def __init__(
        self, 
        model,
        # v_net,
        buffer_size=1000,
        batch_size=100,
        actor_learn_freq=1,
        target_update_freq=5,
        target_update_tau=0.01,
        learning_rate=3e-3,
        discount_factor=0.99,
        gae_lamda=1,
        verbose=False,
        update_iteration=10,
        use_priority=False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._gae_lamda = gae_lamda
        self._target = target_update_freq > 0
        self._update_iteration = update_iteration
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size) # off-policy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = model.policy_net.to(self.device).train()
        self.critic_eval = model.value_net.to(self.device).train()
        self.value_eval = model.v_net.to(self.device).train()

        self.value_target = self.copy_net(self.value_eval)
        
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.value_eval_optim = optim.Adam(self.value_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()

    def learn(self):
        pg_loss, v_loss, q_loss = 0, 0, 0
        for _ in range(self._update_iteration):
            batch_split = self.buffer.split_batch(self._batch_size)
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device)
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
            M = torch.tensor(batch_split['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch_split['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)

            new_A, log_prob = self.actor_eval.evaluate(S)
            
            # V_value loss
            new_q1_value, new_q2_value = self.critic_eval(S, new_A)
            next_value = torch.min(new_q1_value, new_q2_value) - log_prob
            value = self.value_eval(S)
            value_loss = self.criterion(value, next_value.detach())

            # Soft q loss
            q1_value, q2_value = self.critic_eval(S, A)
            target_value = self.value_target(S_)
            target_q_value = R + M * self._gamma * target_value.cpu()
            target_q_value = target_q_value.to(self.device)
            q_value_loss = 0.5 * (self.criterion(q1_value, target_q_value.detach()) + self.criterion(q2_value, target_q_value.detach()))

            # policy loss
            policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()
            # policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value).detach()).mean()

            # update V
            self.value_eval_optim.zero_grad()
            value_loss.backward()
            self.value_eval_optim.step()

            # update soft Q
            self.critic_eval_optim.zero_grad()
            q_value_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            # update policy
            self.actor_eval_optim.zero_grad()
            policy_loss.backward()
            self.actor_eval_optim.step()

            pg_loss += policy_loss.item()
            v_loss += value_loss.item()
            q_loss += q_value_loss.item()

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.value_target, self.value_eval, self.tau)
                
            return pg_loss, q_loss, v_loss
