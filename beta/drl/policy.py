import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

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
        save_path = os.path.join(save_dir, save_file_name)
        actor_save = save_path + '_actor.pth'
        critic_save = save_path + '_critic.pth'
        torch.save(self.actor_eval.state_dict(), actor_save)
        torch.save(self.critic_eval.state_dict(), critic_save)
        print (f'Save actor-critic model in {save_path}!')

    def load_model(self, save_dir, save_file_name, test=False):
        save_path = os.path.join(save_dir, save_file_name)
        actor_save = save_path + '_actor.pth'
        assert os.path.exists(actor_save), f'No {actor_save} file to load'
        self.actor_eval.load_state_dict(torch.load(actor_save))
        print (f'Loading actor model success!')
        if test: return
        critic_save = save_path + '_critic.pth'
        assert os.path.exists(critic_save), f'No {critic_save} file to load'
        self.critic_eval.load_state_dict(torch.load(critic_save))
        print (f'Loading critic model success!')

    @staticmethod
    def soft_sync_weight(target, source, tau=0.01):
        with torch.no_grad():
            for target_param, eval_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def GAE(rewards, v_evals, next_v_eval=0, masks=None, gamma=0.99, lam=1): # [r1, r2, ..., rT], [V1, V2, ... ,VT, VT+1]
        '''
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

        :param gamma: (float) Discount factor
        :param lam: (float) GAE factor
        '''
        # GAE(gamma, lam=0) = td_error
        # GAE(gamma, lam=1) = MC
        assert isinstance(rewards, np.ndarray), 'rewards must be np.ndarray'
        assert isinstance(v_evals, np.ndarray), 'v_evals must be np.ndarray'
        assert len(rewards) == len(v_evals), 'V_pred length must equal rewards length'

        rew_len = len(rewards)
        masks = np.ones(rew_len) if not masks else masks
        v_evals = np.append(v_evals, next_v_eval)
        adv_gae = np.empty(rew_len, 'float32')
        lastgaelam = 0
        for i in reversed(range(rew_len)):
            nonterminal = masks[i]
            delta = rewards[i] + gamma * v_evals[i+1] * nonterminal - v_evals[i]
            adv_gae[i] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        return adv_gae


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
        gae_lamda=1,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.gae_lamda = gae_lamda
        self.save_data = {'log_probs':[], 'values':[], 'rewards':[], 'masks': []}

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

        self.save_data['log_probs'].append(log_prob)
        self.save_data['values'].append(state_value)

        return action.item()

    def learn(self):
        next_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device)
        critic = self.critic_target if self._target else self.critic_eval
        assert len(self.save_data['values']) == len(self.save_data['rewards']), "Error: not same size"

        with torch.no_grad():
            rewards = torch.tensor(self.save_data['rewards']).numpy()
            v_evals = torch.tensor(self.save_data['values']).numpy()
            adv_gae_mc = self.GAE(rewards, v_evals, next_v_eval=0, masks=self.save_data['masks'], gamma=self._gamma, lam=1) # MC adv
            advantage = torch.from_numpy(adv_gae_mc).to(self.device).reshape(1, -1)

        v_eval = torch.stack(self.save_data['values']).reshape(1, -1) # values = torch.stack(save_values, dim=1)
        v_target = advantage + v_eval.detach()
        # critic_core
        critic_loss = self.criterion(v_eval, v_target)

        self.critic_eval.train()
        self.critic_eval_optim.zero_grad()
        critic_loss.backward()
        self.critic_eval_optim.step()
        self._learn_cnt += 1

        if self._learn_cnt % self.actor_learn_freq == 0:
            log_probs = torch.stack(self.save_data['log_probs']).unsqueeze(0) # [1, len(...)]
            # actor_core
            actor_loss = (-log_probs * advantage).sum()
            self.actor_eval.train()
            self.actor_eval_optim.zero_grad()
            actor_loss.backward()
            self.actor_eval_optim.step()

        if self._target:
            if self._learn_cnt % self.target_update_freq == 0:
                if self._verbose: print (f'=======Soft_sync_weight of AC=======')
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        
        self.save_data = {'log_probs':[], 'values':[], 'rewards':[], 'masks': []}

    def sample(self, env, max_steps, test=False):
        assert env, 'You must set env for sample'
        rewards = 0
        state = env.reset()
        for step in range(max_steps):
            action = self.choose_action(state, test)

            next_state, reward, done, info = env.step(action)
            env.render()
            # process env callback
            self.process(s=state, a=action, s_=next_state, r=reward, d=done, i=info)
            rewards += reward

            if done:
                state = env.reset()
                break
            state = next_state
        self.next_state = state
        if self._verbose: print (f'------End eps at {step} steps------')

        return rewards

    def process(self, **kwargs):
        reward, done = kwargs['r'], kwargs['d']

        mask = 0 if done else 1
        reward = torch.tensor(reward, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        self.save_data['rewards'].append(reward)
        self.save_data['masks'].append(mask)


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
        self._batch_size = batch_size
        self.replay_buffer = buffer
        assert buffer.allow_replay, 'DDPG buffer must be replay buffer'

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
            memory_batch = self.replay_buffer.random_sample(self._batch_size)
            batch_split = self.replay_buffer.split(memory_batch)
            
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device) # [batch_size, S.feature_size]
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)
            # d = torch.tensor(done, dtype=torch.float32, device=self.device) # ?
            R = torch.tensor(batch_split['r'], dtype=torch.float32, device=self.device).unsqueeze(-1)

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
            self.process(s=state, a=action, s_=next_state, r=reward, d=done)
            reward_avg += reward
            # process env callback
            if done:
                state = env.reset()
                break
            state = next_state
        if self._verbose: print (f'------End eps at {step} steps------')

        return reward_avg/(step+1)

    def process(self, **kwargs):
        # state, action, next_state, reward = kwargs['s'], kwargs['a'], kwargs['s_'], kwargs['r']
        self.replay_buffer.append(**kwargs)


class PPOPolicy(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        buffer=None,
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
        self.ratio_clip = 0.25
        # self.save_data = {'s':[], 'a':[], 's_':[], 'r':[], 'l':[]}

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
        self._batch_size = batch_size
        self.buffer = buffer
        assert not buffer.allow_replay, 'PPO buffer cannot be replay buffer'

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
        # dist = self.actor_eval(state)
        # m = Categorical(dist)
        # action = m.sample()
        # log_prob = m.log_prob(action)
        # if test:
        #     self.actor_eval.eval()

        # with torch.no_grad():
        #     mu, sigma = self.actor_eval(state)        
        # dist = Normal(mu, sigma)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)
        # return action.item(), log_prob.item()
        return action.item(), log_prob.item()

    def learn(self):
        if not self.buffer.is_full(): 
            print (f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0
        loss_actor_avg = 0
        loss_critic_avg = 0
        actor_cnt = 0

        memory_split = self.buffer.split(self.buffer.all_memory())
        all_S = torch.tensor(memory_split['s'], dtype=torch.float32, device=self.device)
        all_A = torch.tensor(memory_split['a'], dtype=torch.float32, device=self.device)
        all_S_ = torch.tensor(memory_split['s_'], dtype=torch.float32, device=self.device)
        all_R = torch.tensor(memory_split['r'], dtype=torch.float32, device=self.device).unsqueeze(-1)
        all_Log = torch.tensor(memory_split['l'], dtype=torch.float32, device=self.device).unsqueeze(-1)

        rewards = all_R.cpu().numpy()
        with torch.no_grad():
            v_evals = self.critic_eval(all_S).cpu().numpy()
            end_s_ = torch.tensor(memory_split['s_'][-1], dtype=torch.float32, device=self.device)
            end_v_eval = self.critic_eval(end_s_).cpu().numpy()

        adv_gae_td = self.GAE(rewards, v_evals, next_v_eval=end_v_eval, gamma=self._gamma, lam=0) # td_error adv
        advantage = torch.from_numpy(adv_gae_td).to(self.device).unsqueeze(-1)

        # print (f'Size: S:{all_S.size()}, R:{all_R.size()}, advantage:{advantage.size()}, Log: {all_Log.size()}')

        for _ in range(self._update_iteration):
            batch_size = min(len(self.buffer), self._batch_size)
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

            S = all_S[indices]
            A = all_A[indices]
            S_ = all_S_[indices]
            R = all_R[indices].reshape(-1, 1)
            Log = all_Log[indices]
            # print (f'Size: S:{S.size()}, A:{A.size()}, S_:{S_.size()}, R: {R.size()}, Log:{Log.size()}')

            # critic_core
            with torch.no_grad():
                v_target = self.critic_eval(S_)
                if self._target:
                    v_target = self.critic_target(S_)
                v_target = R + self._gamma * v_target
            v_eval = self.critic_eval(S)

            critic_loss = self.criterion(v_eval, v_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_cnt += 1
            
            if self._learn_cnt % self.actor_learn_freq == 0:
                # actor_core
                
                # mu, sigma = self.actor_eval(S)
                # dist = Normal(mu, sigma)
                # new_log_prob = dist.log_prob(A)
                
                dist = self.actor_eval(S)
                m = Categorical(dist)
                action = m.sample()
                new_log_prob = m.log_prob(action)
                new_log_prob = new_log_prob.reshape(-1, 1)

                ratio = torch.exp(new_log_prob - Log) # size = [batch_size, 1]
                # print (f'new {new_log_prob.size()}, old {Log.size()}')
                # print (f'ratio-1 {ratio-1}')
                # print (f'ratio clip {ratio.clamp(0.9, 1.1)}')
                # print (f'ratio size {ratio.size()}')
                # surrogate objective of TRPO
                L1 = advantage[indices] * ratio 
                L2 = advantage[indices] * ratio.clamp(1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                print (f'Size L1 {L1.size()}, L2 {L2.size()}')
                Clipped_surrogate_objective_function = -torch.min(L1, L2).mean()
                # policy entropy
                loss_entropy = torch.mean(torch.exp(new_log_prob) * new_log_prob)
                lambda_entropy = 0

                actor_loss = Clipped_surrogate_objective_function + lambda_entropy * loss_entropy
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
        
        self.buffer.clear()
        print (f'critic_cnt {self._learn_cnt}, actor_cnt {actor_cnt}')
        loss_actor_avg /= actor_cnt
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg

    def sample(self, env, max_steps, test=False):
        assert env, 'You must set env for sample'
        reward_avg = 0
        state = env.reset()
        for step in range(max_steps):
            action, log_prob = self.choose_action(state, test)
            # action = np.clip(action, -1, 1)
            # action_max = env.action_space.high[0]
            # print (action*action_max)

            next_state, reward, done, info = env.step(action)
            env.render()
            # process env callback
            self.process(s=state, a=action, s_=next_state, r=reward, l=log_prob)
            # for process test
            # if len(self.save_data['s']) < len(self.buffer):
            #     self.save_data['s'].append(state)
            #     self.save_data['a'].append(action)
            #     self.save_data['s_'].append(next_state)
            #     self.save_data['r'].append(reward)
            #     self.save_data['l'].append(log_prob)

            reward_avg += reward

            if done:
                state = env.reset()
                break
            state = next_state
        self.next_state = state
        if self._verbose: print (f'------End eps at {step} steps------')

        return reward_avg

    def process(self, **kwargs):
        self.buffer.append(**kwargs)