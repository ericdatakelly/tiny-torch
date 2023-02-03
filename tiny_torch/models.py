from torch import optim
from torchvision import models


def setup_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    return fn(num_classes=10)


def setup_optim(name, params, lr):
    if name in optim.__dict__:
        fn = optim.__dict__[name]
    else:
        raise RuntimeError(f"Unknown optimizer name {name}")

    return fn(params, lr=lr)
