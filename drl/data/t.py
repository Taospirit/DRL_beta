import os, sys
o_path = os.getcwd()

s = o_path
# s = '/'.join(o_path.split('/')[:-1])
sys.path.append(s)

# print (o_path)
print (s)
print (type(o_path))
from data import batch
from data import buffer
# sys.path.append(o_path)