import numpy as np
import random
from collections import namedtuple
# from batch import Batch

class ReplayBuffer(object):
    def __init__(self, size, replay=True):
        self._maxsize = size
        self.memory = []
        self.memory_tuple = namedtuple('memory_tuple', ['S', 'A', 'S_', 'R'])
        self.append_index = 0
        self.allow_replay = replay

    def __len__(self):
        return len(self.memory)

    def append(self, **kwargs):
        if not self.allow_replay and self.is_full():
            return 
        # memory_tuple = Batch(**kwargs)
        memory_tuple = kwargs
        if len(self.memory) < self._maxsize:
            self.memory.append(None)
        self.memory[self.append_index] = memory_tuple
        self.append_index = (self.append_index + 1) % self._maxsize

    def append_(self, state, action, next_state, reward):
        # memory_tuple = Batch(S=state, A=action, R=reward, S_=state_)
        memory_tuple = self.memory_tuple(state, action, next_state, reward)
        if len(self.memory) < self._maxsize:
            self.memory.append(None)
        self.memory[self.append_index] = memory_tuple
        self.append_index = (self.append_index + 1) % self._maxsize

    def random_sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def clear(self):
        self.memory = []
        self.append_index = 0
        print (f'Clear buffer size of {self._maxsize}!') 

    def show(self):
        print (self.memory)

    def split(self, batchs):
        ans = {}
        keys = batchs[-1].keys()
        for key in keys:
            values = [item[key] for item in batchs]
            ans[key] = values

        return ans

    def is_full(self):
        return len(self.memory) == self._maxsize

    def capacity(self):
        return self._maxsize

    def all_memory(self):
        return self.memory

# test = ReplayBuffer(5)
# for i in range(5):
#     test.append(i=i, ans=[1, i], cc=f'{i}: c')

# # test.show()
# sam = test.random_sample(2)
# print (sam)
# ans = test.split(sam)
# print (ans)
# B = test.random_sample(2)
# for b in B:
#     # print (b)
#     print (b.S, b.A, b.R, b.S_)

# c = test.split(B)
# print (c)