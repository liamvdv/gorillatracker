from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np
import torch.ao.quantization
import wandb
from fsspec import Callback
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from print_on_steroids import logger
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import allow_exported_model_train_eval  # type: ignore
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from dlib import get_rank  # type: ignore
from gorillatracker.args import TrainingArgs
from gorillatracker.data.nlet import NletDataModule
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from gorillatracker.model import BaseModule
from gorillatracker.quantization.utils import get_model_input
from gorillatracker.utils.train import ModelConstructor
from gorillatracker.utils.wandb_logger import WandbLoggingModule


def train_and_validate_model(
    args: TrainingArgs,
    dm: NletDataModule,
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
        deterministic=(
            args.force_deterministic if args.force_deterministic and not args.use_quantization_aware_training else False
        ),
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
        wandb.log({"trainer/global_step": 0})  # HACK: to make sure the global_step is logged before the validation
        trainer.validate(model, dm)
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
    dm: NletDataModule,
    model_cls: Type[BaseModule],
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
    wandb_logging_module: WandbLoggingModule,
    embeddings_logger_callback: LogEmbeddingsToWandbCallback,
) -> Trainer:
    # TODO(memben):!!! Fix kfold_k

    dataloader_name = dm.get_dataset_class_names()[0]
    current_process_rank = get_rank()
    kfold_k = int(str(args.data_dir).split("-")[-1])

    # Inject kfold_k into the datamodule TODO(memben): is there a better way?
    dm.kwargs["k"] = kfold_k

    for val_i in range(kfold_k):
        logger.info(f"Rank {current_process_rank} | k-fold iteration {val_i+1} / {kfold_k}")

        # Inject val_i into the datamodule  TODO(memben): is there a better way?
        dm.kwargs["val_i"] = val_i

        kfold_prefix = f"fold-{val_i}"

        embeddings_logger_callback.kfold_k = val_i
        model_constructor = ModelConstructor(args, model_cls, dm)
        model_kfold = model_constructor.construct(wandb_logging_module, wandb_logger)
        model_kfold.kfold_k = val_i

        early_stopping_callback = EarlyStopping(
            monitor=f"{dataloader_name}/{kfold_prefix}/val/loss",
            mode="min",
            min_delta=args.min_delta,
            patience=args.early_stopping_patience,
        )

        checkpoint_callback = ModelCheckpoint(
            filename="snap-{epoch}-samples-loss-{val/loss:.2f}",
            monitor=f"{dataloader_name}/{kfold_prefix}/val/loss",
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=int(args.save_interval),
        )

        _, trainer = train_and_validate_model(
            args,
            dm,
            model_kfold,
            [checkpoint_callback, *callbacks, early_stopping_callback],
            wandb_logger,
        )

    if args.kfold and not args.fast_dev_run:
        kfold_averaging(wandb_logger)

    return trainer  # TODO(rob2u): why return a single model?


def train_using_quantization_aware_training(
    args: TrainingArgs,
    dm: NletDataModule,
    model: BaseModule,
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
    checkpoint_callback: ModelCheckpoint,
) -> Tuple[BaseModule, Trainer]:
    logger.info("Preperation for quantization aware training...")
    example_inputs, _ = get_model_input(dm.dataset_class, args.data_dir, amount_of_tensors=100)  # type: ignore
    example_inputs = (example_inputs,)  # type: ignore
    autograd_graph = capture_pre_autograd_graph(model.model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())  # type: ignore
    model.model = prepare_qat_pt2e(autograd_graph, quantizer)

    allow_exported_model_train_eval(model.model)

    torch.use_deterministic_algorithms(True, warn_only=True)
    model, trainer = train_and_validate_model(args, dm, model, callbacks, wandb_logger)

    logger.info("Quantizing model...")
    quantized_model = convert_pt2e(model.model)
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)
    logger.info("Quantization finished! Saving quantized model...")
    assert checkpoint_callback.dirpath is not None
    save_path = str(Path(checkpoint_callback.dirpath) / "quantized_model_dict.pth")
    torch.save(quantized_model.state_dict(), save_path)

    if args.save_model_to_wandb:
        logger.info("Saving quantized model to wandb...")
        artifact = wandb.Artifact(name=f"quantized_model-{wandb_logger.experiment.id}", type="model")
        artifact.add_file(save_path, name="quantized_model_dict.pth")

    return model, trainer


def kfold_averaging(wandb_logger: WandbLogger) -> None:
    run_path = wandb_logger.experiment.path
    read_access_run = wandb.Api().run(run_path)  # type: ignore

    summary = read_access_run.summary

    metrics = [(key, value) for key, value in summary.items() if "/val/embeddings" in key]

    aggregated_metrics = defaultdict(list)

    # Step 1: Extract metrics by fold and group them
    for key, value in metrics:
        if isinstance(value, (int, float)):
            base_keys = key.split("/")
            base_key: str
            if "fold" in base_keys[1]:
                base_key = f"{base_keys[0]}/{'/'.join(base_keys[2:])}"  # remove fold from key
            else:
                continue  # skip metrics that do not fit the pattern
            aggregated_metrics[f"aggregated/{base_key}"].append(value)

    # Step 2: Compute averages
    average_metrics = {}
    for key, values in aggregated_metrics.items():
        if "pca" in key or "tsne" in key:  # skip pca and tsne metrics as we cannot average them
            continue
        average_metrics[key] = np.mean(values)

    wandb.log(average_metrics, commit=True)


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
