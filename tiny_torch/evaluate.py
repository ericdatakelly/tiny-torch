from pathlib import Path

import torch
from data import setup_data
from models import setup_model
from utils import setup_parser


def evaluate(model_path):
    config = setup_parser().parse_args()

    _, dataloader_eval = setup_data(config)

    model = setup_model(config.model)
    model.load_state_dict(torch.load(model_path))

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader_eval:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network : %d %%' % (100 * correct / total))
