# tiny-torch

A small algorithm for exploring MLOps best practices.

[![Code-Generator](https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700)](https://github.com/pytorch-ignite/code-generator)

This is the image classification template by Code-Generator using `resnet18` model and `cifar10` dataset from TorchVision and training is powered by PyTorch and PyTorch-Ignite.


## Usage

### Run on single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python tiny_torch/main.py configs/resnet152.yaml
```

### Run on single node and multiple GPUs

```bash
torchrun --nproc_per_node=2 tiny_torch/main.py configs/resnet152.yaml --backend=nccl
```


#### Computational profiling with Tensorboard

```bash
torchrun --nproc_per_node=2 tiny_torch/main.py configs/profile_resnet152.yaml --backend=nccl
```

### Monitor model training with Tensorboard

```bash
tensorboard --port=8667 --logdir=tiny-torch/logs
```
