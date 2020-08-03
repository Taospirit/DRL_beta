import numpy as np
import random
from collections import namedtuple


class ReplayBuffer(object):
    def __init__(self, size):
        self._maxsize = size
        self.memory = []
        self.memory_tuple = namedtuple('memory_tuple', ['S', 'A', 'S_', 'R'])
        self.append_index = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, next_state, reward):
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

    def show(self):
        print (self.memory)

    def split(self, batchs):
        # assert isinstance(batchs[-1], Batch)
        # keys = vars(batchs[-1]).keys()
        # State, Action, Reward, State_ = [], [], [], []
        Batch_S = [item.S for item in batchs]
        Batch_A = [item.A for item in batchs]
        Batch_S_ = [item.S_ for item in batchs]
        Batch_R = [item.R for item in batchs]

        return Batch_S, Batch_A, Batch_S_, Batch_R

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