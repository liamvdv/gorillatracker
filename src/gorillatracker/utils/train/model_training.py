from pathlib import Path
from typing import Tuple, Union

import torch.ao.quantization
import wandb
from fsspec import Callback
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from print_on_steroids import logger
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from dlib import get_rank  # type: ignore
from gorillatracker.args import TrainingArgs
from gorillatracker.data_modules import NletDataModule
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from gorillatracker.model import BaseModule
from gorillatracker.quantization.utils import get_model_input
from gorillatracker.ssl_pipeline.data_module import SSLDataModule


def train_and_validate_model(
    args: TrainingArgs,
    dm: Union[SSLDataModule, NletDataModule],
    model: BaseModule,
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
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
        logger.info(f"Rank {current_process_rank} | max_epochs: {model.max_epochs}")

        # TODO(Emirhan): this model is currently NOT being saved, we should save it after each fold
        model, trainer = train_and_validate_model(args, dm, model, callbacks, wandb_logger)

        dm.val_fold = i  # type: ignore
        embeddings_logger_callback.kfold_k = i

    return model, trainer


def train_using_quantization_aware_training(
    args: TrainingArgs,
    dm: Union[SSLDataModule, NletDataModule],
    model: BaseModule,
    callbacks: list[Callback],
    wandb_logger: WandbLogger,
    checkpoint_callback: ModelCheckpoint,
) -> Tuple[BaseModule, Trainer]:
    logger.info("Preperation for quantization aware training...")
    example_inputs, _ = get_model_input(dm.dataset_class, str(args.data_dir), amount_of_tensors=100)  # type: ignore
    example_inputs = (example_inputs,)  # type: ignore
    model.model = capture_pre_autograd_graph(model.model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())  # type: ignore
    model.model = prepare_qat_pt2e(model.model, quantizer)  # type: ignore

    torch.ao.quantization.allow_exported_model_train_eval(model.model)

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
