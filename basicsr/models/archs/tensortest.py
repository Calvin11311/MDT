import torch
import numpy as np
# bs=2
# D = torch.full((H, 3,3), bs).transpose(1,0)
# print(D)
# dist_func = lambda x: x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (1 + torch.outer(D[0], x**2).flatten() + torch.outer(D[1], x**4).flatten() + torch.outer(D[2], x**6).flatten() +torch.outer(D[3], x**8).flatten())
# theta_max = dist_func(torch.tensor([1]).cuda())

# a = torch.tensor([1])
# dist_func = lambda x: x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (1 + torch.outer(D[0], x**2).flatten() + torch.outer(D[1], x**4).flatten() + torch.outer(D[2], x**6).flatten() +torch.outer(D[3], x**8).flatten())
# theta_max = dist_func(torch.tensor([1]))
# x=torch.tensor([3,3,3,3])
# print(x.repeat_interleave(3, 0).flatten().shape)

# theta_max1=x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten()
# print(theta_max1)
# print(x**4)
# theta_max2=(1 + torch.outer(D[0], x**2).flatten() + torch.outer(D[1], x**4).flatten() 
#             + torch.outer(D[2], x**6).flatten() +torch.outer(D[3], x**8).flatten())
# print(theta_max2)
# theta_max=theta_max1*theta_max2
# print(theta_max)
# print(theta_max.repeat_interleave(D.shape[1],0))


