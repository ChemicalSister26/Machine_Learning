import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    # data loader
    def __init__(self):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    # dataset[index]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # len(dataset)
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

# first_part = dataset[0]
# features, labels = first_part
# print(features)
# print(labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# it = iter(dataloader)
# it1 = it.next()
# features, labels = it1
# print(features)
# print(labels)

# training loop

epochs = 2
total_samples = len(dataset)
number_of_iterations = math.ceil(total_samples/4)

for i in range(epochs):
    for j, (input, labels) in enumerate(dataloader):
        if i % 5 == 0:
            print(i, j, input.shape)


