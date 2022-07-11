import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    # data loader
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    # dataset[index]
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    # len(dataset)
    def __len__(self):
        return self.n_samples

class Totensor:
    def __call__(self, sample):
        inputs, targets = sample

        return torch.from_numpy(inputs), torch.from_numpy(targets)

class Multiple:
    def __init__(self, factor):
        self.factor = factor
    def __multiple__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


# dataset = WineDataset(transform=Totensor())
# first_batch = dataset[0]
# features, labels = first_batch
# print(type(features), type(labels))

composed = torchvision.transforms.Compose(Totensor(), Multiple(3))
dataset = WineDataset(composed)
first_batch = dataset[0]
features, labels = first_batch
print(type(features), type(labels))



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
#
# epochs = 2
# total_samples = len(dataset)
# number_of_iterations = math.ceil(total_samples/4)
#
# for i in range(epochs):
#     for j, (input, labels) in enumerate(dataloader):
#         if i % 5 == 0:
#             print(i, j, input.shape)
#

