import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from policy import BasicPolicy


class DDPGPolicy(BasePolicy):
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        buffer,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        batch_size=100,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

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
        loss_actor_avg = 0
        loss_critic_avg = 0
        actor_cnt = 0

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
                actor_cnt += 1
                if self._verbose: print (f'=======Learn_Actort_Net=======')

            if self._target:
                if self._learn_cnt % self.target_update_freq == 0:
                    if self._verbose: print (f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                    self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        
        loss_actor_avg /= actor_cnt
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg

    def sample(self, env, max_steps, test=False):
        assert env, 'You must set env for sample'
        reward_avg = 0
        state = env.reset()
        for step in range(max_steps):
            action = self.choose_action(state, test)
            action = action.clip(-1, 1)
            action_max = env.action_space.high[0]

            next_state, reward, done, info = env.step(action * action_max)
            env.render()
            self.process(s=state, a=action, s_=next_state, r=reward, d=done, i=info)
            reward_avg += reward
            # process env callback
            if done:
                state = env.reset()
                break
            state = next_state
        if self._verbose: print (f'------End eps at {step} steps------')

        return reward_avg/(step+1)

    def process(self, **kwargs):
        state, action, next_state, reward = kwargs['s'], kwargs['a'], kwargs['s_'], kwargs['r']
        self._buffer.append(state, action, next_state, reward)