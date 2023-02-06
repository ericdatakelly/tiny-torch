from pprint import pformat
from typing import Any

import ignite.distributed as idist
import yaml
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed
from torch import nn

from tiny_torch.data import setup_data
from tiny_torch.models import setup_model, setup_optim
from tiny_torch.trainers import setup_evaluator, setup_trainer
from tiny_torch.utils import *


def run(local_rank: int, config: Any):

    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder
    config.output_dir = setup_output_dir(config, rank)

    # donwload datasets and create dataloaders
    dataloader_train, dataloader_eval = setup_data(config)

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config.model))
    optimizer = idist.auto_optim(
        setup_optim(config.optimizer, model.parameters(), lr=config.lr)
    )
    loss_fn = nn.CrossEntropyLoss().to(device=device)

    # trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, loss_fn, device, dataloader_train.sampler
    )
    evaluator = setup_evaluator(config, model, device)

    # attach metrics to evaluator
    accuracy = Accuracy(device=device)
    metrics = {
        "eval_accuracy": accuracy,
        "eval_loss": Loss(loss_fn, device=device),
        "eval_error": (1.0 - accuracy) * 100,
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # setup engines logger with python logging
    # print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(vars(config)))
    (config.output_dir / "config-lock.yaml").write_text(yaml.dump(config))
    trainer.logger = evaluator.logger = logger

    # setup ignite handlers
    to_save_train = {"model": model, "optimizer": optimizer, "trainer": trainer}
    to_save_eval = {"model": model}
    ckpt_handler_train, ckpt_handler_eval = setup_handlers(
        trainer, evaluator, config, to_save_train, to_save_eval
    )
    # experiment tracking
    if rank == 0:
        tb_logger = setup_exp_logging(config, trainer, optimizer, evaluator)

    # print metrics to the stderr
    # with `add_event_handler` API
    # for training stats
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    prof = None
    if getattr(config, "profile", False):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    if prof is not None:
        # run evaluation at every training epoch end
        # with shortcut `on` decorator API and
        # print metrics to the stderr
        # again with `add_event_handler` API
        # for evaluation stats
        @trainer.on(Events.EPOCH_COMPLETED(every=2))
        def _():
            evaluator.run(dataloader_eval)
            log_metrics(evaluator, "eval")

    # let's try to run evaluation first as a sanity check
    @trainer.on(Events.STARTED)
    def _():
        evaluator.run(dataloader_eval)

    if prof is not None:
        prof.start()

        @trainer.on(Events.ITERATION_COMPLETED)
        def _():
            prof.step()

    # setup is done. let's run the training
    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
    )

    if prof is not None:
        prof.stop()

    # close logger
    if rank == 0:
        tb_logger.close()

    # show last checkpoint names
    logger.info(
        "Last training checkpoint name - %s",
        ckpt_handler_train.last_checkpoint,
    )

    logger.info(
        "Last evaluation checkpoint name - %s",
        ckpt_handler_eval.last_checkpoint,
    )


# main entrypoint
def main():
    config = setup_config()
    with idist.Parallel(config.backend) as p:
        p.run(run, config=config)


if __name__ == "__main__":
    main()
