import torch
import numpy as np

'''
Refer: https://github.com/thu-ml/tianshou
'''

class Batch(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        '''return self[index]'''
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None: # 有值就更新返回
                b.__dict__.update(**{k: self.__dict__[k][index]})
        return b
    #TODO:
    def __len__(self):
        pass

    def append(self):
        pass

    def split(self):
        pass