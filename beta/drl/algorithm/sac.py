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

#TODO
class SAC(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        gae_lamda=1,
        verbose = False
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
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device)
        self.critic_eval = critic_net.to(self.device)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = self.copy_net(self.actor_eval)
            self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.SmoothL1Loss()

    def choose_action(self, state, test=False):
        # state = torch.tensor(state, dtype=torch.float32, device=self.device)
        # if test:
        #     self.actor_eval.eval()
        #     return Categorical(self.actor_eval(state)).sample().item(), 0
        
        # dist = self.actor_eval(state)
        # m = Categorical(dist)
        # action = m.sample()
        # log_prob = m.log_prob(action)
        # state_value = self.critic_eval(state)

        # return action.item(), log_prob

    def learn(self):
        memory_split = self.buffer.split(self.buffer.all_memory()) # s, a, r, m, s_
        S = torch.tensor(memory_split['s'], dtype=torch.float32, device=self.device)
        A = torch.tensor(memory_split['a'], dtype=torch.float32).view(-1, 1)
        R = torch.tensor(memory_split['r'], dtype=torch.float32).view(-1, 1)
        M = torch.tensor(memory_split['m'], dtype=torch.float32).view(-1, 1)
        S_ = torch.tensor(memory_split['s_'], dtype=torch.float32, device=self.device)
    
        # v_eval = self.critic_eval(S)

        # v_evals = v_eval.detach().cpu().numpy()
        # rewards = R.numpy()
        # masks = M.numpy()
        # adv_gae_mc = self.GAE(rewards, v_evals, next_v_eval=0, masks=masks, gamma=self._gamma, lam=self._gae_lamda) # MC adv
        # advantage = torch.from_numpy(adv_gae_mc).to(self.device).reshape(-1, 1)

        # v_target = advantage + v_eval.detach()
        # # critic_core
        # critic_loss = self.criterion(v_eval, v_target)
        # self.critic_eval_optim.zero_grad()
        # critic_loss.backward()
        # self.critic_eval_optim.step()
        # self._learn_critic_cnt += 1

        # if self._learn_critic_cnt % self.actor_learn_freq == 0:
        #     # actor_core
        #     actor_loss = (-Log * advantage).sum()
        #     self.actor_eval.train()
        #     self.actor_eval_optim.zero_grad()
        #     actor_loss.backward()
        #     self.actor_eval_optim.step()
        #     self._learn_actor_cnt += 1

        # if self._target:
        #     if self._learn_critic_cnt % self.target_update_freq == 0:
        #         if self._verbose: print (f'=======Soft_sync_weight of AC=======')
        #         self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
        #         self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        #         self._sync_cnt += 1
        
        self.buffer.clear()
        assert self.buffer.is_empty()
