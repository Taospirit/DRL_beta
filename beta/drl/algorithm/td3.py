import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer


class TD3(BasePolicy):
    def __init__(
        self,
        model,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=1,
        target_update_tau=1,
        learning_rate=3e-3,
        discount_factor=0.99,
        batch_size=100,
        verbose=False,
    ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._update_iteration = 10
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_eval = model.policy_net.to(self.device).train()
        self.critic_eval = model.value_net.to(self.device).train()

        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.MSELoss()  # why mse?

        self.noise_clip = 0.5
        self.noise_std = 0.2

    def learn(self):
        loss_actor_avg, loss_critic_avg = 0, 0

        for _ in range(self._update_iteration):
            batch_split = self.buffer.split_batch(self._batch_size)
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device)  # [batch_size, S.feature_size]
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
            M = torch.tensor(batch_split['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch_split['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)
            # print (f'SIZE S {S.size()}, A {A.size()}, M {M.size()}, R {R.size()}, S_ {S_.size()}')

            A_noise = self.actor_target.action(S_, self.noise_std, self.noise_clip)
            with torch.no_grad():
                q1_next, q2_next = self.critic_target.twinQ(S_, A_noise)
                q_next = torch.min(q1_next, q2_next)
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(self.device)

            q1_eval, q2_eval = self.critic_eval.twinQ(S, A)
            loss1 = self.criterion(q1_eval, q_target)
            loss2 = self.criterion(q2_eval, q_target)
            critic_loss = 0.5 * (loss1 + loss2)

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()

            loss_critic_avg += 0.5 * critic_loss.item()
            self._learn_critic_cnt += 1

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                actor_loss = -self.critic_eval(S, self.actor_eval(S)).mean()
                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1
                if self._verbose:
                    print(f'=======Learn_Actort_Net=======')

            if self._learn_critic_cnt % self.target_update_freq == 0:
                if self._verbose:
                    print(f'=======Soft_sync_weight of DDPG=======')
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration
        return loss_actor_avg, loss_critic_avg
