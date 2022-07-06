import os
import torch

from torch import nn
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

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#forward pass
def forward(x):
     return w*x
#loss
def loss(y, y_pred):
   return(((y_pred-y)**2).mean())

print(f'Prediction before training {forward(5)}')

#training

learning_rate = 0.01
epoch_number = 1000

for epoch in range(epoch_number):
    #forward calculations
    y_prediction = forward(X)
    #loss
    l = loss(Y, y_prediction)

    #gradients = backward pass dl/dw
    l.backward()

    #update weights
    with torch.no_grad():
        w -= learning_rate*w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'weight {w:3f}, loss {l:.8f}')

print(f'Prediction after training {forward(5)}')