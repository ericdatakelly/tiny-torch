from pathlib import Path

import torch

from dataset import create_loader
from model import Net


def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network : %d %%' % (100 * correct / total))


testloader = create_loader(train=False)

model_path = Path('output', 'model.pth')
model = Net()
model.load_state_dict(torch.load(model_path))

evaluate(model, testloader)
