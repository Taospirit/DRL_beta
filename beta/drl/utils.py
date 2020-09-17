import numpy as np
import random
from collections import namedtuple

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
            # values = [item[key] for item in batchs]
            values = np.array([item[key] for item in batchs])
            # print (type(values))
            split_res[key] = values
            # print (split_res[key].shape)
            # for item in batchs:
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

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
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

    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data # sefl.data store data
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return self.sum_tree[index], data_index, index
        # pi^a, data_index, sum_tree_index

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'mask'))
class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, size, replay=True):
        super().__init__()
        self.capacity = size
        self.history = 4
        self.discount = 0.99
        self.n = 3
        self.priority_weight = 0.4
        self.priority_exponent = 0.5
        self.t = 0
        self.SumTree = SegmentTree(capacity)

    def append(self, state, action, reward, mask):
        self.SumTree.append(Transition(self.t, state, action, reward, mask), self.SumTree.max)
        self.t = 0 if mask self.t + 1
       
    def _get_transition(self, idx): # n-step transition
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.SumTree.get(idx)
        for t in range(self.history - 2, -1, -1):
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans
            else:
                transition[t] = self.SumTree.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.SumTree.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans
        return transition

    def _get_sample_from_segment(self, segment, i):
        valid = False
        # to be check
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, idx, tree_idx = self.SumTree.find(sample)
            if (self.SumTree.index - idx) % self.capacity > self.n and (idx - self.SumTree.index) % self.capacity >= self.history and prob != 0:
                valid = True

        transition = self._get_transition(idx)
        states = [trans.state for trans in transition[:self.history]]
        next_states = [trans.state for trans in transition[self.n:self.n + self.history]]
        actions = [trans.action for trans in transition[:self.history]]
        discounted_rewards = [sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))]
        mask = [transition[self.history + self.n - 1].nonterminal]

        return prob, idx, tree_idx, states, actions, discounted_rewards, next_states, mask
    # todo
    # def sample(self, batch_size):
    #     p_total = self.SumTree.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    #     segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    #     batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
    #     probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    #     states, next_states, = torch.stack(states), torch.stack(next_states)
    #     actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
    #     probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    #     capacity = self.capacity if self.SumTree.full else self.SumTree.index
    #     weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    #     weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    #     return tree_idxs, states, actions, returns, next_states, nonterminals, weights


    # def update_priorities(self, idxs, priorities):
    #     priorities = np.power(priorities, self.priority_exponent)
    #     [self.SumTree.update(idx, priority) for idx, priority in zip(idxs, priorities)]



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