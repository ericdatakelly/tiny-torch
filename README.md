# tiny-torch

A small algorithm for exploring MLOps best practices.

[![Code-Generator](https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700)](https://github.com/pytorch-ignite/code-generator)

This is the image classification template by Code-Generator using `resnet18` model and `cifar10` dataset from TorchVision and training is powered by PyTorch and PyTorch-Ignite.

## Installation

Navigate to the `tiny-torch` directory, create a conda environment, and activate it.

```bash
conda env create -f environment.yaml
```
Install the library with pip.

```bash
pip install -e .
```

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

Be sure `torch_tb_profiler` is installed in the environment and use the profiling config file.

```bash
torchrun --nproc_per_node=2 tiny_torch/main.py configs/profile_resnet152.yaml --backend=nccl
```
From the `tiny-torch` directory, start Tensorboard.

```bash
tensorboard --port=8667 --logdir=logs
```
In Nebari, navigate to `https://<your nebari domain>/user/<your username>/proxy/8667/` (include the trailing slash) to view Tensorboard.  Click on the PYTORCH_PROFILER tab.
