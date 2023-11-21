import dataclasses
import importlib
import os
import time
from pathlib import Path

import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import parse

from args import TrainingArgs
from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank, wait_for_debugger
from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule
from gorillatracker.helpers import check_checkpoint_path_for_wandb, check_for_wandb_checkpoint_and_download_if_necessary
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from model import get_model_cls

WANDB_PROJECT = "MNIST-EfficientNetV2"
WANDB_ENTITY = "gorillas"

def get_data_module_class(module: str, online: bool = True):
    parent = QuadletDataModule if online else TripletDataModule
    mod = importlib.import_module(module)
    for symbol_name in dir(mod):
        v = getattr(mod, symbol_name)
        # Check if the symbol is a subclass of the given class
        if issubclass(v, parent) and v is not parent:
            print(f"Selected DataModule: {v}")
            return v

def main(args: TrainingArgs):
    ########### CUDA checks ###########
    current_process_rank = get_rank()
    logger.config(rank=current_process_rank, print_rank0_only=True)
    if args.accelerator == "cuda":
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus > args.num_devices:
            logger.warning(
                f"Requested {args.num_devices} GPUs but {num_available_gpus} are available.",
                f"Using first {args.num_devices} GPUs. You should set CUDA_VISIBLE_DEVICES or the docker --gpus flag to the desired GPU ids.",
            )
        if not torch.cuda.is_available():
            logger.error("CUDA is not available, you should change the accelerator with --accelerator cpu|tpu|mps.")
            exit(1)
    if current_process_rank == 0 and args.debug:
        wait_for_debugger()  # TODO: look into debugger usage

    args.seed = seed_everything(workers=True, seed=args.seed)

    ############# Construct W&B Logger ##############
    if args.offline or args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(name=args.run_name)

    if args.saved_checkpoint_path and args.resume and check_checkpoint_path_for_wandb(args.saved_checkpoint_path):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(args.saved_checkpoint_path), resume="must"
        )  # resume W&B run

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=args.wandb_tags,
        save_dir="logs/",
        **wandb_extra_args,
    )
    wandb_logger.log_hyperparams(dataclasses.asdict(args))
    # wandb_logger.experiment.log_code(".")  # log code to wandb to be able to reproduce the run

    if current_process_rank == 0:
        logger.info(args)
    if current_process_rank == 0 and not args.resume and not args.offline:
        if args.run_name is None:
            logger.warning("No run name specified with `--run_name`. Using W&B default (randomly generated name).")
        else:
            assert wandb_logger.version is not None
            # wandb_logger.experiment.name = (
            #     args.run_name + "-" + wandb_logger.version
            # )  # Append id to name for easier recognition in W&B UI
            wandb_logger.experiment.name = args.run_name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    ################# Construct model ##############

    # Resume from checkpoint if specified
    model_args = dict(
        model_name_or_path=args.model_name_or_path,
        from_scratch=args.from_scratch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        lr_decay=args.lr_decay,
        lr_decay_interval=args.lr_decay_interval,
        margin=args.margin,
        loss_mode=args.loss_mode,
    )
    model_cls = get_model_cls(args.model_name_or_path)

    if args.saved_checkpoint_path is None:
        args.saved_checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.saved_checkpoint_path, wandb_logger.experiment
        )

        if args.resume:  # load weights, optimizer states, scheduler state, ...\
            model = model_cls.load_from_checkpoint(args.saved_checkpoint_path, save_hyperparameters=False)
            # we will resume via trainer.fit(ckpt_path=...)
        else:  # load only weights
            model = model_cls(**model_args)
            torch_load = torch.load(args.saved_checkpoint_path, map_location=torch.device(model_args["accelerator"]))
            model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = model_cls(**model_args)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
        model = torch.compile(model)

    #################### Construct dataloaders & trainer #################
    dm_cls = get_data_module_class(args.data_module, online=args.loss_mode.startswith("online"))
    dm = dm_cls.from_training_args(args)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    embeddings_logger_callback = LogEmbeddingsToWandbCallback(
        every_n_val_epochs=args.embedding_save_interval,
        wandb_run=wandb_logger.experiment,
    )

    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{epoch}-samples-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        every_n_epochs=int(args.save_interval),
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.early_stopping_patience,
    )

    callbacks = [
        checkpoint_callback,
        wandb_disk_cleanup_callback,
        lr_monitor,
        early_stopping,
        embeddings_logger_callback,
    ]
    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.num_devices,
        accelerator=args.accelerator,
        strategy=args.distributed_strategy,
        logger=wandb_logger,
        deterministic=args.force_deterministic,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        # accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=args.fast_dev_run,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
    )

    if current_process_rank == 0:
        logger.info(
            f"Total optimizer epochs: {args.max_epochs} | "
            f"Model Log Frequency: {args.save_interval} | "
            f"Effective batch size: {args.batch_size} | "
        )

    ########### Start val & train loop ###########
    if args.val_before_training and not args.resume:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        if args.only_val:
            exit(0)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, dm, ckpt_path=args.saved_checkpoint_path if args.resume else None)

    if trainer.interrupted:
        logger.warning("Detected keyboard interrupt, trying to save latest checkpoint...")
    else:
        logger.success("Fit complete")

    if current_process_rank == 0:
        logger.info("Trying to save checkpoint....")

        save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
        trainer.save_checkpoint(save_path)

        logger.info("Collecting PL checkpoint for wandb...")
        artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
        artifact.add_file(save_path, name="model.ckpt")

        logger.info("Pushing to wandb...")
        aliases = ["train_end", "latest"]
        wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

        logger.success("Saving finished!")


if __name__ == "__main__":
    print("Starting training script...")
    config_path = "./cfgs/config.yml"
    print("Parsing config from {}...")
    parsed_arg_groups = parse(TrainingArgs, config_path=config_path)
    # parses the config file as default and overwrites with command line arguments
    # therefore allowing sweeps to overwrite the defaults in config file
    current_process_rank = get_rank()
    with graceful_exceptions(extra_message=f"Rank: {current_process_rank}"):
        main(parsed_arg_groups)
