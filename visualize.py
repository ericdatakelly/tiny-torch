import matplotlib.pyplot as plt
import numpy as np
import torchvision

from data import setup_data
from utils import setup_parser


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def view_examples():
    config = setup_parser().parse_args()

    _, dataloader_eval = setup_data(config)

    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    )

    # get some random training images
    dataiter = iter(dataloader_eval)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
