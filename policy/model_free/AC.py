from Base import BasePolicy
from data import Batch
from copy import deepcopy
import torch
import torch.nn as nn

'''
Refer: https://github.com/thu-ml/tianshou
Refer: https://github.com/Yonv1943/DL_RL_Zoo
'''

class BasicActorCritic(BasePolicy): 
    '''tricks
    @ double network for overcome overestimation
    @ use advantage for actor-net policy gradient update
    '''
    def __init__(self, actor_net, critic_net, discount_factor=0.99, target_update_freq=0, **_kwargs):
        '''argument init'''
        self.learning_rate = 4e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_freq = target_update_freq
        self.policy_noise = policy_noise
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._sync_cnt = 0
        self.action_dist_fn = torch.distributions.Categorical

        '''net_work init'''
        self.actor_eval = actor_net
        self.actor_eval.train()
        self.actor_eval_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=self.learning_rate)

        self.critic_eval = critic_net
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

    def learn(self, buffer, batch_size, max_steps, repeat_times):
        self.actor_eval.eval()
        
        loss_actor_sum, loss_critic_sum = 0, 0
        
        # Bathc(S=state, A=action, R=reward, G=gamma, S_=next_state, D=distribution)
        for i in range(max_steps * repeat_times):
            Batchs = buffer.random_sample(batch_size)
            # for batch in batchs.split(batch_size): # 单独一个state
            with torch.no_grad():
                V_next = self.critic_eval(Batchs.S_)
                if self._target:
                    V_next = self.critic_target(Batchs.S_)

                V_target = reward + gamma * V_next
                V_eval = self.critic_eval(Batchs.S)

            # critic loss
            loss_critic = self.criterion(V_eval, V_target)
            loss_critic_sum += loss_critic.item()
            
            self.critic_eval_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_eval_optimizer.step()
            
            # actor loss
            if i % repeat_times == 0:
                # on-policy
                with torch.no_grad():
                    Advantage = V_target - self.critic_eval(Batchs.S)# TD-error
                    if self._target:
                        Advantage = V_target - self.critic_target(Batchs.S)
                #FIXME to be mixed
                loss_actor = -(Batchs.D.log_prob(Batchs.A) * Advantage).mean()
                loss_actor_sum += loss_actor

                self.actor_eval_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_eval_optimizer.step()

            '''soft target update'''
            #FIXME:
            self._sync_cnt += 1
            if self._sync_cnt == self.target_update_freq:
                self.soft_sync_weight(self.actor_target, self.actor_eval)
                self.soft_sync_weight(self.critic_target, self.critic_eval)
                self._sync_cnt = 0


    def choose_action(self, states, explore_noise=0.0):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        logits = self.actor_eval(states).cpu().data.numpy()
        actions = self.action_dist_fn(logits).sample()
        return actions
        
    def soft_sync_weight(target, eval, tau=5e-3):
        for target_param, eval_param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)