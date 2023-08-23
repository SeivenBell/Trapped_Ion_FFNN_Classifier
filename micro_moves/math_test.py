import torch

a = torch.rand(3)
b = torch.rand(5)
print(a, end="\r\n")
print(b, end="\r\n")
print(str(torch.einsum("i,j->ij", a,b)))
