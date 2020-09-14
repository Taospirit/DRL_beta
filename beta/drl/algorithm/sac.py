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
        actor_net, 
        critic_net,
        v_net,
        buffer_size=1000,
        batch_size=100,
        actor_learn_freq=1,
        target_update_freq=5,
        target_update_tau=0.01,
        learning_rate=3e-3,
        discount_factor=0.99,
        gae_lamda=1,
        verbose=False,
        action_space=None
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
        self._update_iteration = 10
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size) # off-policy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device).train()
        self.critic_eval = critic_net.to(self.device).train()
        self.value_eval = v_net.to(self.device).train()

        self.value_target = self.copy_net(self.value_eval)
        # self.actor_target = self.copy_net(self.actor_eval)
        # self.critic_target = self.copy_net(self.critic_eval)
        
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.value_eval_optim = optim.Adam(self.value_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        # self.action_max = action_space.high[0]
        action_scale = (action_space.high - action_space.low) / 2
        action_bias = (action_space.high + action_space.low) / 2
        # self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        # self.action_bias = torch.tensor(action_bias, dtype=torch.float32, device=self.device)
      
        # self.action_num = action_space.shape[0]

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = 0.2
        # self.alpha = self.log_alpha.exp()
        # self.target_entropy = -torch.Tensor(self.n_actions).to(self.device) 

    def choose_action_(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        mean, log_std = self.actor_eval(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        # print (f'device {y_t.device}, scale {self.action_scale.device}, bias {self.action_bias.device}')
        action = y_t * self.action_scale + self.action_bias

        # sample = Normal(mean, std).rsample()
        # action = torch.tanh(sample).detach()
        return action.detach().cpu().numpy()[0]

    def choose_action(self, state, test=False):
        # state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor_eval.action(state)
        # action = action * self.action_scale + self.action_bias
        action = action * 2 + 0
        return action

    # Use re-parameterization tick
    def get_action_log_prob(self, model, state):
        # epsilon = 1e-6
        mean, log_std = model(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # x_t = dist.rsample()
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias

        # log_prob = dist.log_prob(x_t)
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        noise = Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = dist.log_prob(mean + std * z.to(self.device))
        log_prob -= torch.log(self.action_scale*(1 - action.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
    
    def learn_(self):
        for _ in range(self._update_iteration):
            batch_split = self.buffer.split_batch(self._batch_size)
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device)
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
            M = torch.tensor(batch_split['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch_split['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)
            # Log = torch.tensor(batch_split['l'], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                # A_, Log_, _ = self.actor_eval.sample(S_) #TODO:
                A_, Log_ = self.get_action_log_prob(self.actor_target, S_)
                q1_next, q2_next = self.critic_target(S_, A_) # ?(S, A_)
                q_next = torch.min(q1_next, q2_next) - self.alpha * Log_
                # q_next = self.critic_target(S_, A_) - self.alpha * Log_
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(self.device)
            
            q1_eval, q2_eval = self.critic_eval(S, A)
            critic_loss = self.criterion(q1_eval, q_target) + self.criterion(q2_eval, q_target)

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            # critic loss
            # q_eval = self.critic_eval(S, A)
            # critic_loss = self.criterion(q_eval, q_target)
            # self.critic_eval_optim.zero_grad()
            # critic_loss.backward()
            # self._learn_critic_cnt += 1
            
            if self._learn_critic_cnt % self.actor_learn_freq:
                # a_curr, log_curr, _ = self.actor_eval.sample(S) #TODO:
                A, Log = self.get_action_log_prob(self.actor_eval, S)
                q1_next, q2_next = self.critic_eval(S, A)
                q_next = torch.min(q1_next, q2_next)
                # actor loss
                # actor_loss = (self.alpha * Log - q_next.detach()).mean()
                actor_loss = (self.alpha * Log - q_next).mean()
                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()

                # alpha loss
                alpha_loss = -(self.log_alpha * (Log + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

    def learn(self):
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
            q_loss = self.criterion(q1_value, target_q_value.detach()) + self.criterion(q2_value, target_q_value.detach())

            # policy loss
            policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()
            # policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value).detach()).mean()

            # update V
            self.value_eval_optim.zero_grad()
            value_loss.backward()
            self.value_eval_optim.step()

            # update soft Q
            self.critic_eval_optim.zero_grad()
            q_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            # update policy
            self.actor_eval_optim.zero_grad()
            policy_loss.backward()
            self.actor_eval_optim.step()

            if self._learn_critic_cnt % self.target_update_freq:
                # self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.value_target, self.value_eval, self.tau)