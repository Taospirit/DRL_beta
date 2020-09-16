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


class PPO(BasePolicy):  # option: double
    def __init__(
        self,
        model,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.0001,
        discount_factor=0.99,
        gae_lamda=0,  # td
        batch_size=100,
        verbose=False
    ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.ratio_clip = 0.2
        self.lam_entropy = 0.01
        self.adv_norm = False  # normalize advantage, defalut=False
        self.rew_norm = False  # normalize reward, default=False
        self.schedule_clip = False
        self.schedule_adam = False

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._gae_lam = gae_lamda
        self._target = target_update_freq > 0
        self._update_iteration = 10
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

        self._verbose = verbose
        self._batch_size = batch_size
        self._normalized = lambda x, e: (x - x.mean()) / (x.std() + e)
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = model.policy_net.to(self.device).train()
        self.critic_eval = model.value_net.to(self.device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        # self.actor_eval.train()
        # self.critic_eval.train()

        if self._target:
            self.actor_target = self.copy_net(self.actor_eval)
            self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.SmoothL1Loss()

    # def choose_action(self, state, test=False):
    #     state = torch.tensor(state, dtype=torch.float32, device=self.device)
    #     # if test:
    #     #     self.actor_eval.eval()
    #     # with torch.no_grad():
    #     #     mu, sigma = self.actor_eval(state)
    #     # dist = Normal(mu, sigma)
    #     # action = dist.sample()
    #     # # print (f'mu:{mu}, sigma:{sigma}, dist: {dist}, action sample before clamp: {action}')
    #     # action = action.clamp(-2, 2)
    #     # # print (f'action after clamp {action}')
    #     # log_prob = dist.log_prob(action)
    #     # assert abs(action.item()) <= 2, f'ERROR: action out of {action}'
    #     # return action.item(), log_prob.item()
    #     return self.actor_eval.action(state)

    # # not use
    # def get_batchs_indices(self, buffer_size, batch_size, replace=True, batch_num=None):
    #     indices = [i for i in range(buffer_size)]
    #     if replace: # 有放回的采样
    #         if not batch_num:
    #             batch_num = round(buffer_size / batch_size + 0.5) * 2
    #         return [np.random.choice(indices, batch_size, replace=False) for _ in range(batch_num)]
    #     else:# 无放回的采样
    #         np.random.shuffle(indices)
    #         return [indices[i: i + batch_size] for i in range(0, buffer_size, batch_size)]

    def learn(self, i_episode=0, num_episode=100):
        if not self.buffer.is_full():
            print(f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0

        loss_actor_avg = 0
        loss_critic_avg = 0

        memory_split = self.buffer.split(self.buffer.all_memory())
        S = torch.tensor(memory_split['s'], dtype=torch.float32, device=self.device)
        A = torch.tensor(memory_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
        S_ = torch.tensor(memory_split['s_'], dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_split['r'], dtype=torch.float32).view(-1, 1)
        Log = torch.tensor(memory_split['l'], dtype=torch.float32, device=self.device).view(-1, 1)

        # print (f'Size S {S.size()}, A {A.size()}, S_ {S_.size()}, R {R.size()}, Log {Log.size()}')
        # print (f'S {S}, A {A}, S_ {S_}, R {R}, Log {Log}')
        with torch.no_grad():
            v_evals = self.critic_eval(S).cpu().numpy()
            end_v_eval = self.critic_eval(S_[-1]).cpu().numpy()

        rewards = self._normalized(R, self.eps).numpy() if self.rew_norm else R.numpy()
        # rewards = rewards.cpu().numpy()
        adv_gae_td = self.GAE(rewards, v_evals, next_v_eval=end_v_eval,
                              gamma=self._gamma, lam=self._gae_lam)  # td_error adv
        advantage = torch.from_numpy(adv_gae_td).to(self.device).unsqueeze(-1)
        advantage = self._normalized(advantage, 1e-10) if self.adv_norm else advantage

        # indices = [i for i in range(len(self.buffer))]
        for _ in range(self._update_iteration):
            v_eval = self.critic_eval(S)
            v_target = advantage + v_eval.detach()

            critic_loss = self.criterion(v_eval, v_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                # actor_core
                mu, sigma = self.actor_eval(S)
                dist = Normal(mu, sigma)
                new_log_prob = dist.log_prob(A)

                pg_ratio = torch.exp(new_log_prob - Log)  # size = [batch_size, 1]
                clipped_pg_ratio = torch.clamp(pg_ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                surrogate_loss = -torch.min(pg_ratio * advantage, clipped_pg_ratio * advantage).mean()

                # policy entropy
                loss_entropy = -torch.mean(torch.exp(new_log_prob) * new_log_prob)

                actor_loss = surrogate_loss - self.lam_entropy * loss_entropy

                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1
                if self._verbose:
                    print(f'=======Learn_Actort_Net=======')

            if self._target:
                if self._learn_critic_cnt % self.target_update_freq == 0:
                    if self._verbose:
                        print(f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                    self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

        self.buffer.clear()
        assert self.buffer.is_empty()

        # update param
        ep_ratio = 1 - (i_episode / num_episode)
        if self.schedule_clip:
            self.ratio_clip = 0.2 * ep_ratio

        if self.schedule_adam:
            new_lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        print(f'critic_cnt {self._learn_critic_cnt}, actor_cnt {self._learn_actor_cnt}')
        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg
