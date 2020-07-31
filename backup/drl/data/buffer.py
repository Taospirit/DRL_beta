import numpy as np
import random
# try:
from drl.data import Batch
# except ImportError:
#     print ('NOT find pkg in pip, load local pkg')
    

class Buffer(object):
    def __init__(self, size):
        self._maxsize = size
        self.memory = []
        self.append_index = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, state_):
        memory_tuple = Batch(S=state, A=action, R=reward, S_=state_)
        # memory_tuple = [state, action, reward, state_]
        if len(self.memory) < self._maxsize:
            self.memory.append(None)
        self.memory[self.append_index] = memory_tuple
        self.append_index = (self.append_index + 1) % self._maxsize

    def random_sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    def clear(self):
        # if self.memory: print (f'Buffer to be cleared, size is {len(self.memory)}')
        self.memory = []
        self.append_index = 0

    def show(self):
        print (self.memory)

    def split(self, batchs):
        assert isinstance(batchs[-1], Batch)
        # keys = vars(batchs[-1]).keys()
        # State, Action, Reward, State_ = [], [], [], []
        States = [item.S for item in batchs]
        Actions = [item.A for item in batchs]
        Rewards = [item.R for item in batchs]
        State_s = [item.S_ for item in batchs]

        return States, Actions, Rewards, State_s

    def is_full(self):
        return len(self.memory) == self._maxsize

# test = Buffer(3)
# for i in range(5):
#     test.append(i, 22, 33, 44)

# B = test.random_sample(2)
# for b in B:
#     # print (b)
#     print (b.S, b.A, b.R, b.S_)

# c = test.split(B)
# print (c)