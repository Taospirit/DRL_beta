import torch

a = torch.randn(2, 2)
# print (a)
# print (a.mean())
# print (a.mean().item())
b = torch.Tensor([1, 2, 3])
c = b*b
print (c)
d = torch.Tensor([2, 3, 4])
print (c*d)

print(torch.tensor([0.99]*10))