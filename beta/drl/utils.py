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

    def clear(self):
        self.memory = []
        self.append_index = 0
        print (f'----------Clear buffer size of {self._maxsize}!----------') 

    def show(self):
        print (self.memory)

    def split(self, batchs):
        split_res = {}
        for key in batchs[-1].keys():
            values = [item[key] for item in batchs]
            split_res[key] = values

        return split_res

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
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

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