from typing import Any, Union

import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events
from ignite.metrics import Frequency
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler


def setup_trainer(
    config: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    device: Union[str, torch.device],
    train_sampler: Sampler,
) -> Union[Engine, DeterministicEngine]:
    def train_function(engine: Union[Engine, DeterministicEngine], batch: Any):
        model.train()

        samples = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        with autocast(config.use_amp):
            outputs = model(samples)
            loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss = loss.item()

        engine.state.metrics = {
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"]
        }
        return {"train_loss": train_loss, "num_samples": len(samples)}

    trainer = Engine(train_function)

    throughput_metric = Frequency(output_transform=lambda x: x['num_samples'])
    throughput_metric.attach(trainer, name='throughput')

    # set epoch for distributed sampler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)

    return trainer


def setup_evaluator(
    config: Any,
    model: Module,
    device: Union[str, torch.device],
) -> Engine:
    @torch.no_grad()
    def eval_function(engine: Engine, batch: Any):
        model.eval()

        samples = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        with autocast(config.use_amp):
            outputs = model(samples)

        return outputs, targets

    return Engine(eval_function)
