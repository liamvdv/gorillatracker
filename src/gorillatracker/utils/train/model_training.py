from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import wandb
from fsspec import Callback
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from print_on_steroids import logger

from dlib import get_rank  # type: ignore
from gorillatracker.args import TrainingArgs
from gorillatracker.data_modules import NletDataModule
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from gorillatracker.model import BaseModule
from gorillatracker.ssl_pipeline.data_module import SSLDataModule


def train_and_validate_model(
    args: TrainingArgs,
    dm: Union[SSLDataModule, NletDataModule],
    model: BaseModule,
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
    model_name_suffix: Optional[str] = "",
) -> Tuple[BaseModule, Trainer]:
    trainer = Trainer(
        num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        devices=args.num_devices,
        accelerator=args.accelerator,
        strategy=str(args.distributed_strategy),
        logger=wandb_logger,
        deterministic=args.force_deterministic,
        callbacks=callbacks,
        precision=args.precision,  # type: ignore
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=24,
        # accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=args.fast_dev_run,
        profiler=args.profiler,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
        plugins=args.plugins,  # type: ignore
        # reload_dataloaders_every_n_epochs=1,
    )

    ########### Start val & train loop ###########
    if args.val_before_training and not args.resume:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.
        logger.info("Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        if args.only_val:
            return model, trainer
    logger.info("Starting training...")
    trainer.fit(model, dm, ckpt_path=args.saved_checkpoint_path if args.resume else None)

    if trainer.interrupted:
        logger.warning("Detected keyboard interrupt, trying to save latest checkpoint...")
    else:
        logger.success("Fit complete")

    ########### Save checkpoint ###########
    current_process_rank = get_rank()

    if current_process_rank == 0:
        save_model(args, callbacks[0], wandb_logger, trainer, model_name_suffix)
    else:
        logger.info("Rank is not 0, skipping checkpoint saving...")

    return model, trainer


def train_and_validate_using_kfold(
    args: TrainingArgs,
    dm: Union[SSLDataModule, NletDataModule],
    model: BaseModule,
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
    embeddings_logger_callback: LogEmbeddingsToWandbCallback,
) -> Tuple[BaseModule, Trainer]:

    current_process_rank = get_rank()
    kfold_k = int(str(args.data_dir).split("-")[-1])
    dm.k = kfold_k  # type: ignore

    for i in range(kfold_k):
        logger.info(f"Rank {current_process_rank} | k-fold iteration {i+1} / {kfold_k}")

        model, trainer = train_and_validate_model(args, dm, model, callbacks, wandb_logger, f"_fold_{i}")

        dm.val_fold = i  # type: ignore
        embeddings_logger_callback.kfold_k = i

    if args.kfold and not args.fast_dev_run:
        kfold_averaging(wandb_logger)

    return model, trainer


def kfold_averaging(wandb_logger: WandbLogger) -> None:
    run_path = wandb_logger.experiment.path
    read_access_run = wandb.Api().run(run_path)  # type: ignore

    summary = read_access_run.summary

    metrics = [(key, value) for key, value in summary.items() if "val/embeddings/fold" in key]

    aggregated_metrics = defaultdict(list)

    # Step 1: Extract metrics by fold and group them
    for key, value in metrics:
        if isinstance(value, (int, float)):
            base_key = "/".join(key.split("/")[:2] + ["averaged"] + key.split("/")[3:])  # Remove fold part
            aggregated_metrics[base_key].append(value)

    # Step 2: Compute averages
    average_metrics = {}
    for key, values in aggregated_metrics.items():
        average_metrics[key] = np.mean(values)

    wandb.log(average_metrics)


def save_model(
    args: TrainingArgs,
    checkpoint_callback: ModelCheckpoint,
    wandb_logger: WandbLogger,
    trainer: Trainer,
    name_suffix: Optional[str] = "",
) -> None:
    logger.info("Trying to save checkpoint....")

    assert checkpoint_callback.dirpath is not None
    save_path = str(Path(checkpoint_callback.dirpath) / f"last_model_ckpt{name_suffix}.ckpt")
    trainer.save_checkpoint(save_path)
    logger.info(f"Checkpoint saved to {save_path}")

    if args.save_model_to_wandb:
        logger.info("Collecting PL checkpoint for wandb...")
        artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}{name_suffix}", type="model")
        artifact.add_file(save_path, name="model.ckpt")

        logger.info("Pushing to wandb...")
        aliases = ["train_end", "latest", name_suffix]
        wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

        logger.success("Saving finished!")
