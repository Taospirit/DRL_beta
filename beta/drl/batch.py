import numpy as np

class Batch(object):

    def __init__(self, **kwargs):
        super().__init__()
        # print (kwargs)
        self.__dict__.update(kwargs)

    def split(self):
        pass

b = Batch(a=4, b=[1, 1], c='111')
print (b.a)
