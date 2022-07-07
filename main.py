import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


#preparing data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

number_samples, number_features = x.shape

# create model

input_size = number_features
output_size = 1

model = nn.Linear(input_size, output_size)

# loss and optimizer
learning_rate = 0.01
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#traininhg loop

epoch_number = 100
for epoch in range(epoch_number):
    #forward calculations
    y_prediction = model(x)
    #loss
    loss = criteria(y_prediction, y)

    #gradients = backward pass dl/dw
    loss.backward()

    #update weights
    optimizer.step()

     # zero gradients
    optimizer.zero_grad()


if epoch % 10 == 0:
    [w, b] = model.parameters()
    print(f'loss {loss.item():.8f}')

predicted = model(x).detach().numpy()
plt.plot(x, y, 'ro')
plt.plot(x, predicted, 'g')
plt.show()

