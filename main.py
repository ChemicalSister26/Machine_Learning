import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

# import numpy as np
# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)
#
# w = 0.0
# #forward pass
# def forward(x):
#     return w*x
#
# #loss
# def loss(y, y_pred):
#   return(((y_pred-y)**2).mean())
#
# #gradient
# def grad(x,y,y_pred):
#     return np.dot(2*x, y_pred-y).mean()
#
# print(f'Prediction before training {forward(5)}')
#
# #training
#
# learning_rate = 0.01
# epoch_number = 20
#
# for epoch in range(epoch_number):
#     #forward calculations
#     y_prediction = forward(X)
#     #loss
#     l = loss(Y, y_prediction)
#
#     #gradients
#     dw = grad(X, Y, y_prediction)
#
#     #update weights
#     w -= learning_rate*dw
#
#     if epoch % 1 == 0:
#         print(f'weight {w:3f}, loss {l:.8f}')
#
# print(f'Prediction after training {forward(5)}')


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

number_samples, number_features = X.shape
input_size = number_features
output_size = number_features



class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

    #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)


print(f'Prediction before training {model(X_test).item()}')

#training

learning_rate = 0.01
epoch_number = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_number):
    #forward calculations
    y_prediction = model(X)
    #loss
    l = loss(Y, y_prediction)

    #gradients = backward pass dl/dw
    l.backward()

    #update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'weight {w[0][0].item():3f}, loss {l:.8f}')

print(f'Prediction after training {model(X_test).item()}')