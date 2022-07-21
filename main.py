import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# determine device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# determine parameters of full cycle

input_size = 784 #28*28*1
hidden_size = 200
number_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

# preparing of data

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)

examples = iter(train_loader)
samples, labels = examples.next()

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

#creation of model

class NeuralHandwritten(nn.Module):
    def __init__(self, input_size, hidden_size, number_classes):
        super(NeuralHandwritten, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.func = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, number_classes)

    def forward(self, x):
        out = self.l1(x)
        out1 = self.func(out)
        out2 = self.l2(out1)
        return out2

model = NeuralHandwritten(input_size, hidden_size, number_classes)

#loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#create training loop

total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        output = model(images)
        loss = criterion(output, labels)

        #backwared pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch}, step {i}, loss = {loss.item():.4f}')

# evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # torch.max returns value, index - we need only index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100 * n_correct/n_samples
    print(acc)









