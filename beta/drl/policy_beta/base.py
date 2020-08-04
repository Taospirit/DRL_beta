import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

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