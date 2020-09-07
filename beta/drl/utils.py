import numpy as np
import random
# from collections import namedtuple

class ReplayBuffer(object):
    def __init__(self, size, replay=True):
        self._maxsize = size
        self.memory = []
        # self.memory_tuple = namedtuple('memory_tuple', ['S', 'A', 'S_', 'R'])
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

    def random_sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def clear(self): # manual clear
        self.memory = []
        self.append_index = 0
        # print (f'----------Clear buffer size of {self._maxsize}!----------') 

    def show(self):
        print (self.memory)

    def split(self, batchs):
        split_res = {}
        for key in batchs[-1].keys():
            values = [item[key] for item in batchs]
            split_res[key] = values

        return split_res

    def split_batch(self, batch_size):
        return self.split(self.random_sample(batch_size))

    def is_full(self):
        return len(self.memory) == self._maxsize
    
    def is_empty(self):
        return len(self.memory) == 0

    # @property
    def capacity(self):
        return self._maxsize
    
    # @property
    def all_memory(self):
        return self.memory

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__()
    
    def append(self, **kwargs):
        pass

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self._size = size
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)
        self.max = 1

    def _propagate(self, index):
        p = (index - 1) // 2
        l, r = 2 * p + 1, 2 * p + 2
        self.sum_tree[p] = self.sum_tree[l] + self.sum_tree[r]
        if p != 0:
            self._propagate(p)

    def _retrieve(self, index, value):
        l, r = 2 * index + 1, 2 * index + 2
        if l >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[l]:
            return self._retrieve(l, value)
        else:
            return self._retrieve(r, value - self.sum_tree[l])

    def append(self, data, value):
        self.data[self.index] = data # sefl.data store data
        # self.update(self.index + self._size - 1, value)
        tree_index = self.index + self._size - 1
        self.sum_tree[tree_index] = value # self.sum_tree store value
        self._propagate(tree_index)
        # self.max = max(value, self.max)
        self.index = (self.index + 1) % self._sizem
        self.full = self.full or self.index == 0 # ?
        self.max = max(value, self.max)

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return self.sum_tree[index], data_index, index

    def get(self, data_index):
        return self.data[data_index % self._size]

    def total(self):
        return self.sum_tree[0]

class Batch(object):
    def __init__(self, **kwargs):
        super().__init__()
        # print (kwargs)
        self.__dict__.update(kwargs)

    def split(self):
        pass

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape