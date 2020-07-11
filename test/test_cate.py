import torch
import numpy as np
# import torch.distributions.Categorical as Categorical
probs = torch.FloatTensor([[0.05,0.1,0.85],[0.05,0.05,0.9]])

probs = torch.FloatTensor([0.2, 0.4, 0.1, 0.2, 0.1])
dist = torch.distributions.Categorical(probs)
print(dist)
# Categorical(probs: torch.Size([2, 3]))
 
index = dist.sample()
print(index)
# print(index[0])
# [2 2]
print (probs[index.item()])
print (np.log(probs[index.item()]))
ans = dist.log_prob(index)
print (ans)