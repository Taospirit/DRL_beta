import numpy as np
import random
import torch
from collections import namedtuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # values = np.array([item[key] for item in batchs])
            # split_res[key] = values
            split_res[key] = [item[key] for item in batchs]
        # split_res['s_'] = split_res['s'][1:]
        # split_res['s_'].append(self.)
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

    def size(self):
        return self.append_index

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.memory = []
        self.max = 1
        self.cnt = 0

    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    def append(self, data, value):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.index] = data
        self.update(self.index + self.size - 1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            # if self.cnt > 15:
            #     print (f'GET! at {index} for cnt {self.cnt}')
            self.cnt = 0
            return index
        elif value <= self.sum_tree[left]:
            self.cnt += 1
            # if self.cnt > 15:
            #     print (f'search left {left}, size {self.size}')
            # print (f'retrieveing at left {self.cnt}')
            # print (f'left_index {left} to search')
            return self._retrieve(left, value)
        else:
            self.cnt += 1
            # if self.cnt > 15:
            #     print (f'search right {right}, size {self.size}')
            # print (f'retrieveing at right {self.cnt}')
            # print (f'right_index {right} to search')
            return self._retrieve(right, value - self.sum_tree[left])

    def find(self, value):
        tree_index = self._retrieve(0, value)
        data_index = tree_index - self.size + 1
        priority = self.sum_tree[tree_index]
        return (priority, data_index, tree_index)

    def get(self, data_index):
        return self.memory[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
# blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False) # atari game
blank_trans = Transition(0, None, None, 0, False) # gym
class PriorityReplayBuffer(object):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.discount = 0.99
        self.n = 3
        self.priority_weight = 0.4
        self.priority_exponent = 0.5
        self.t = 0
        self.StoreTree = SegmentTree(capacity)
        self.blank_trans = Transition(0, None, None, 0, False)

    def append(self, **kwargs):
        state, action, reward, terminal = kwargs['s'], kwargs['a'], kwargs['r'], not kwargs['m']
        if self.blank_trans.state is None:
            blank_state = np.zeros(np.array(state).shape, dtype=np.float32)
            self.blank_trans = Transition(0, blank_state, None, 0, False)
        self.StoreTree.append(Transition(self.t, state, action, reward, not terminal), self.StoreTree.max)
        self.t = 0 if terminal else self.t + 1

    def _get_transition(self, idx): # n-step transition
        transition = np.array([None] * (self.n + 1))
        transition[0] = self.StoreTree.get(idx)
        for i in range(1, self.n + 1):
            if transition[i - 1].nonterminal:
                transition[i] = self.StoreTree.get(idx + i)
            else:
                transition[i] = blank_trans
        return transition

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.StoreTree.memory))
        def get_sample_from_segment(segment, i):
            valid = False
            cnt = 0
            while not valid:
                sample = np.random.uniform(i * segment, (i + 1) * segment)
                
                cnt += 1
                if cnt > 30:
                    sample = np.random.uniform(0, batch_size * segment)
                prio, idx, tree_idx = self.StoreTree.find(sample)

                forward = (self.StoreTree.index - idx) % self.capacity
                if forward > self.n and prio != 0:
                    valid = True
        
            transition = self._get_transition(idx)
            state = [transition[0].state]
            next_state = [transition[self.n].state]
            action = [transition[0].action]
            discounted_reward = [sum(self.discount ** i * transition[i].reward for i in range(self.n))]
            nonterminal = [transition[self.n].nonterminal]

            return prio, idx, tree_idx, state, action, discounted_reward, next_state, nonterminal

        p_total = self.StoreTree.total()
        segment = p_total / batch_size
        batch = [get_sample_from_segment(segment, i) for i in range(batch_size)]
        prios, idxs, tree_idxs, states, actions, returns, next_states, nonterminal = zip(*batch)

        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        prios = np.array(prios, dtype=np.float32) / p_total
        capacity = self.capacity if self.StoreTree.full else self.StoreTree.index
        weights = (capacity * prios) ** -self.priority_weight

        return tree_idxs, states, actions, returns, next_states, nonterminal, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.StoreTree.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def is_full(self):
        return self.StoreTree.full

    def size(self):
        return self.StoreTree.index

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