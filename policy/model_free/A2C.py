import torch
import torch.nn as nn
from base import BasePolicy
from data import Batch, Buffer
from copy import deepcopy


'''
Refer: https://github.com/thu-ml/tianshou
Refer: https://github.com/Yonv1943/DL_RL_Zoo
'''

class A2C(BasePolicy): 
    '''tricks
    @ double network to overcome overestimation
    @ use advantage for actor-net policy gradient update
    '''
    def __init__(self, actor_net, critic_net, discount_factor=0.99, target_update_freq=0, **_kwargs):
        '''argument init'''
        self.on_policy = True # cannot use replay buffer
        self.learning_rate = 4e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_freq = target_update_freq
        self.policy_noise = policy_noise
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        # self.action_dist_fn = torch.distributions.Categorical

        '''net_work init'''
        self.actor_eval = actor_net.to(self.device) # output is dist
        self.actor_eval.train()
        self.actor_eval_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=self.learning_rate)

        self.critic_eval = critic_net.to(self.device) # output is value
        self.critic_eval.train()
        self.critic_eval_optimizer = torch.optim.Adam(self.critic_eval.optimizer(), lr=self.learning_rate)
        
        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.actor_target.eval()
            self.critic_target = deepcopy(self.critic_eval)
            self.critic_target.eval()

        self.criterion = nn.SmoothL1Loss()

        '''todo'''
        # self.update_counter = 0
    def learn_actor(self, buffer, batch_size):
        self.actor_eval.eval()

        Batchs = buffer.random_sample(batch_size)
        S, A, R, S_ = buffer.split(Batchs)
        gamma = torch.tensor([self._gamma] * batch_size)

        with torch.no_grad():
            V_next = self.critic_eval(S_)
            V = self.critic_eval(S)
            if self._target:
                V_next = self.critic_target(S_)
                V = self.critic_target(S)

            advantage = torch.tensor(R) + gamma * V_next - V # TD-error

        log_probs = [self.actor_eval(b.S).log_prob(b.A) for b in Batchs]
        log_probs = torch.cat(log_probs)
        
        # actor loss
        actor_loss = -(log_probs * advantage.detach()).mean()

        self.actor_eval_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_eval_optimizer.step() # actor network learning

        return actor_loss.item()
        
    def learn_critic(self, buffer, batch_size, nums):
        critic_loss_sum = 0
        for _ in range(nums): # 执行步数为max_step, 轮次为repeat_time
            
            Batchs = buffer.random_sample(batch_size)
            S, A, R, S_ = buffer.split(Batchs)
            gamma = torch.tensor([self._gamma] * batch_size)

            with torch.no_grad():
                V_next = self.critic_eval(S_)
                if self._target:
                    V_next = self.critic_target(S_)
                V_target = torch.tensor(R) + gamma * V_next
                V_eval = self.critic_eval(S)

            # critic loss
            critic_loss = self.criterion(V_eval, V_target)
            critic_loss_sum += critic_loss.item()
            
            self.critic_eval_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_eval_optimizer.step()

        return critic_loss_sum

    def learn(self, buffer, batch_size, actor_learn_freq):
        self.learn_critic(buffer, batch_size, int(len(buffer)/batch_size))
        self._learn_critic_cnt += 1

        if self._learn_critic_cnt % actor_learn_freq == 0:
            self.learn_actor(buffer, batch_size*5)

            if self.on_policy:
                buffer.clear()

        if self._learn_critic_cnt % self.target_update_freq == 0:
            self.soft_sync_weight(self.critic_target, self.critic_eval)
            self._learn_critic_cnt = 0

        if self._learn_actor_cnt % self.target_update_freq == 0:
            self.soft_sync_weight(self.actor_target, self.actor_eval)
            self._learn_actor_cnt = 0

    def learn_(self, buffer, batch_size, max_steps, repeat_times):
        self.actor_eval.eval()
        
        actor_loss_sum, critic_loss_sum = 0, 0
        critic_loss = 0
        actor_loss = 0
        learn_nums = int(len(buffer)/batch_size)
        # Bathc(S=state, A=action, R=reward, S_=next_state, D=distribution)
        for i in range(nums): # 执行步数为max_step, 轮次为repeat_time
            
            Batchs = buffer.random_sample(batch_size)            
            # for item in Batchs: # 单独的<s, a, r, s_>
            #     S, A, R, S_  = item.S, item.A, item.R, item.S_
            #     with torch.no_grad():
            #         V_next = self.critic_eval(S_)
            #         if self._target:
            #             V_next = self.critic_target(S_)

            #         V_target = R + self._gamma * V_next
            #         V_eval = self.critic_eval(S)

            #     critic_loss_item = self.criterion(V_eval, V_target)
            #     critic_loss += critic_loss_item
            S, A, R, S_ = buffer.split(Batchs)


            gamma = torch.tensor([self._gamma] * batch_size)
            with torch.no_grad():
                V_next = self.critic_eval(S_)
                if self._target:
                    V_next = self.critic_target(S_)
                # V_target = reward + gamma * V_next
                V_target = torch.tensor(R) + gamma * V_next
                V_eval = self.critic_eval(S)

            # critic loss
            critic_loss = self.criterion(V_eval, V_target)
            critic_loss_sum += critic_loss.item()
            
            self.critic_eval_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_eval_optimizer.step()
            
            # actor loss
            if i % repeat_times == 0:
                # on-policy
                with torch.no_grad():
                    advantage = V_target - self.critic_eval(S)# TD-error
                    if self._target:
                        advantage = V_target - self.critic_target(S)

                # log_probs = []
                # for b in Batchs:
                #     dist = self.actor_eval(b.S)
                #     log_prob = dist.log_prob(b.A)
                #     log_probs.append(log_prob)

                log_probs = [self.actor_eval(b.S).log_prob(b.A) for b in Batchs]
                log_probs = torch.cat(log_probs)

                #FIXME to be fixed
                actor_loss = -(log_probs * advantage.detach()).mean()               
                actor_loss_sum += actor_loss.item()

                self.actor_eval_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_eval_optimizer.step() # actor network learning

                if self.on_policy:
                    buffer.clear()

            '''soft target update'''
            self._sync_cnt += 1
            if self._sync_cnt == self.target_update_freq:
                self.soft_sync_weight(self.actor_target, self.actor_eval)
                self.soft_sync_weight(self.critic_target, self.critic_eval)
                self._sync_cnt = 0


    def choose_action(self, state, explore_noise=0.0):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = self.actor_eval(state)
        action = dist.sample()
        return action.cpu().numpy()
        
    def soft_sync_weight(target, eval, tau=5e-3):
        for target_param, eval_param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)