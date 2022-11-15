from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_loader
from model import Net

# Get training data
trainloader = create_loader(train=True)

# Instantiate the network, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train
n_epochs = 2
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = batch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save model
output_path = Path('output')
model_path = output_path / 'model.pth'

if not output_path.exists():
    output_path.mkdir()

torch.save(model.state_dict(), model_path)
print(f'Model saved as {model_path}')
