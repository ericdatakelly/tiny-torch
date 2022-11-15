import torch
import torchvision
import torchvision.transforms as transforms


def create_loader(train=True):
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transformer
    )

    print(f'Images in dataset: {len(dataset)}')

    return torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2
    )
