import warnings
from typing import Union

import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.plugins import BitsandbytesPrecision
from print_on_steroids import logger
from simple_parsing import parse
from torchvision.transforms import Compose, Resize

from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank, wait_for_debugger  # type: ignore
from gorillatracker.args import TrainingArgs
from gorillatracker.data_modules import NletDataModule
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from gorillatracker.model import get_model_cls
from gorillatracker.ssl_pipeline.data_module import SSLDataModule
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig
from gorillatracker.train_utils import get_data_module
from gorillatracker.utils.train import (
    ModelConstructor,
    train_and_validate_model,
    train_and_validate_using_kfold,
    train_using_quantization_aware_training,
)
from gorillatracker.utils.wandb_logger import WandbLoggingModule

warnings.filterwarnings("ignore", ".*was configured so validation will run at the end of the training epoch.*")
warnings.filterwarnings("ignore", ".*Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", ".* does not have many workers.*")
warnings.filterwarnings("ignore", ".*site-packages/torchmetrics/utilities/prints.py:43.*")


def main(args: TrainingArgs) -> None:
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
        # if not torch.cuda.is_available():
        #     logger.error("CUDA is not available, you should change the accelerator with --accelerator cpu|tpu|mps.")
        #     exit(1)
    if current_process_rank == 0 and args.debug:
        wait_for_debugger()  # TODO: look into debugger usage

    args.seed = seed_everything(workers=True, seed=args.seed)

    ############# Construct W&B Logger ##############
    wandb_logging_module = WandbLoggingModule(args)
    wandb_logger = wandb_logging_module.construct_logger()

    ################# Construct model class ##############
    model_cls = get_model_cls(args.model_name_or_path)

    #################### Construct dataloaders #################
    model_transforms = model_cls.get_tensor_transforms()
    if args.data_resize_transform is not None:
        model_transforms = Compose([Resize(args.data_resize_transform, antialias=True), model_transforms])

    # TODO(memben): Unify SSLDatamodule and NletDataModule
    dm: Union[SSLDataModule, NletDataModule]
    if args.use_ssl:
        assert args.split_path is not None, "Split path must be provided for SSL training."
        ssl_config = SSLConfig(
            tff_selection=args.tff_selection,
            n_samples=args.n_samples,
            feature_types=args.feature_types,
            min_confidence=args.min_confidence,
            min_images_per_tracking=args.min_images_per_tracking,
            split_path=args.split_path,
        )
        dm = SSLDataModule(
            ssl_config=ssl_config,
            batch_size=args.batch_size,
            transforms=model_transforms,
            training_transforms=model_cls.get_training_transforms(),
            data_dir=str(args.data_dir),
            additional_dataset_class_ids=args.additional_val_dataset_classes,
            additional_data_dirs=args.additional_val_data_dirs,
        )
    else:
        dm = get_data_module(
            args.dataset_class,
            str(args.data_dir),
            args.batch_size,
            args.loss_mode,
            args.workers,
            model_transforms,
            model_cls.get_training_transforms(),
            args.additional_val_dataset_classes,
            args.additional_val_data_dirs,
        )

    ################# Construct model ##############

    model_constructor = ModelConstructor(args, model_cls, dm)
    model = model_constructor.construct(wandb_logging_module, wandb_logger)

    #################### Construct dataloaders & trainer #################
    model_transforms = model.get_tensor_transforms()
    if args.data_resize_transform is not None:
        model_transforms = Compose([Resize(args.data_resize_transform, antialias=True), model_transforms])

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    embeddings_logger_callback = LogEmbeddingsToWandbCallback(
        every_n_val_epochs=args.embedding_save_interval,
        knn_with_train=args.knn_with_train,
        wandb_run=wandb_logger.experiment,
        dm=dm,
        use_quantization_aware_training=args.use_quantization_aware_training,
        use_ssl=args.use_ssl,
    )

    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{epoch}-samples-loss-{val/loss:.2f}",
        monitor="val/loss/dataloader_0",
        mode="min",
        auto_insert_metric_name=False,
        every_n_epochs=int(args.save_interval),
    )

    early_stopping = EarlyStopping(
        monitor="val/loss/dataloader_0",
        mode="min",
        min_delta=args.min_delta,
        patience=args.early_stopping_patience,
    )

    callbacks = (
        [
            checkpoint_callback,  # keep this at the top
            wandb_disk_cleanup_callback,
            lr_monitor,
            early_stopping,
            embeddings_logger_callback,
        ]
        if not args.kfold
        else [
            wandb_disk_cleanup_callback,
            lr_monitor,
            embeddings_logger_callback,
        ]
    )

    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    # Initialize trainer
    supported_quantizations = ["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"]
    if args.precision in supported_quantizations:
        args.plugins = BitsandbytesPrecision(mode=args.precision)  # type: ignore
        args.precision = "16-true"

    if current_process_rank == 0:
        logger.info(
            f"Total optimizer epochs: {args.max_epochs} | "
            f"Model Log Frequency: {args.save_interval} | "
            f"Effective batch size: {args.batch_size} | "
        )

    ################# Start training #################
    logger.info(f"Rank {current_process_rank} | Starting training...")
    if args.kfold:
        train_and_validate_using_kfold(
            args=args,
            dm=dm,
            model_cls=model_cls,
            callbacks=callbacks,
            wandb_logger=wandb_logger,
            wandb_logging_module=wandb_logging_module,
            embeddings_logger_callback=embeddings_logger_callback,
        )
    elif args.use_quantization_aware_training:
        model, trainer = train_using_quantization_aware_training(
            args=args,
            dm=dm,
            model=model,
            callbacks=callbacks,
            wandb_logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
        )
    else:
        train_and_validate_model(args=args, dm=dm, model=model, callbacks=callbacks, wandb_logger=wandb_logger)


if __name__ == "__main__":
    print("Starting training script...")
    config_path = "./cfgs/config.yml"
    parsed_arg_groups = parse(TrainingArgs, config_path=config_path)

    # parses the config file as default and overwrites with command line arguments
    # therefore allowing sweeps to overwrite the defaults in config file
    current_process_rank = get_rank()
    main(parsed_arg_groups)
