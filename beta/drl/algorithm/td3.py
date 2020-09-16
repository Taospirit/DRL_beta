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
        learning_rate=0.01,
        discount_factor=0.99,
        batch_size=100,
        verbose=False,
        action_max=1
    ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        # self._target = target_update_freq > 0
        self._update_iteration = 10
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.actor_eval = model.policy_net.to(self.device).train()
        self.critic_eval = model.value_net.to(self.device).train()

        # self.actor_eval = actor_net.to(self.device).train()
        # self.critic_1 = critic_net.to(self.device)  # two Q net
        # self.critic_2 = deepcopy(critic_net).to(self.device)
        # self.critic_eval = critic_net.to(self.device).train() # CriticQTwin

        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        # self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        # self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        # self.actor_eval.train()
        # self.critic_1.train()
        # self.critic_2.train()

        self.actor_target = self.copy_net(self.actor_eval)
        # self.critic_1_target = self.copy_net(self.critic_1)
        # self.critic_2_target = self.copy_net(self.critic_2)
        self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.MSELoss()  # why mse?

        self.noise_clip = 0.5
        self.noise_std = 0.2
        # self.action_max = action_max

    # def choose_action(self, state, test=False):
    #     if test:
    #         self.actor_eval.eval()
    #     # action = self.actor_eval(state) # out = tanh(x)
    #     # action = action.clamp(-self.action_max, self.action_max)
    #     # return action.item()
    #     return self.actor_eval.predict(state, self.action_max).item()

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

            # noise = torch.ones_like(A).data.normal_(0, self.noise_std).to(self.device)
            # noise = noise.clamp(-self.noise_clip, self.noise_clip)
            # A_noise = self.actor_target(S_) + noise
            # A_noise = A_noise.clamp(-self.action_max, self.action_max)
            A_noise = self.actor_target.predict(S_, self.action_max, self.noise_std, self.noise_clip)
            with torch.no_grad():
                # q1_next = self.critic_1_target(S_, A_noise)  # add noise
                # q2_next = self.critic_2_target(S_, A_noise)
                q1_next, q2_next = self.critic_target.twinQ(S_, A_noise)
                q_next = torch.min(q1_next, q2_next)
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(self.device)

            # q1_eval = self.critic_1(S, A)
            # critic_1_loss = self.criterion(q1_eval, q_target)
            # self.critic_1_optim.zero_grad()
            # critic_1_loss.backward()
            # self.critic_1_optim.step()

            # q2_eval = self.critic_2(S, A)
            # critic_2_loss = self.criterion(q2_eval, q_target)
            # self.critic_2_optim.zero_grad()
            # critic_2_loss.backward()
            # self.critic_2_optim.step()

            # loss_critic_avg += 0.5 * (critic_1_loss.item() + critic_2_loss.item())

            q1_eval, q2_eval = self.critic_eval.twinQ(S, A)
            critic_loss = self.criterion(q1_eval, q_target) + self.criterion(q2_eval, q_target)
            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()

            loss_critic_avg += 0.5 * critic_loss.item()
            self._learn_critic_cnt += 1

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                # actor_loss = -self.critic_1(S, self.actor_eval(S)).mean()  # no noise
                actor_loss = -self.critic_eval(S, self.actor_eval(S)).mean()  # no noise
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
                # self.soft_sync_weight(self.critic_1_target, self.critic_1, self.tau)
                # self.soft_sync_weight(self.critic_2_target, self.critic_2, self.tau)

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration
        return loss_actor_avg, loss_critic_avg
