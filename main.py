import torch
import numpy as np
import math

data = [[1, 2], [3, 4]]
a = torch.tensor(data)
np = np.array(data)
print(np)

b = torch.from_numpy(np)
print(b)

c = torch.ones_like(a)
print(c)
d = torch.rand_like(a, dtype=torch.float)
print(d)

shape = (2, 3,)

s = torch.rand(shape)
e = torch.ones(shape)
f = torch.zeros(shape)

print(s, e, f)