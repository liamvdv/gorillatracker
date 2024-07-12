import warnings
from pathlib import Path

import debugpy
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.plugins import BitsandbytesPrecision
from print_on_steroids import logger
from simple_parsing import parse
from torchvision.transforms import Compose, Normalize, Resize

from gorillatracker.args import TrainingArgs
from gorillatracker.data.builder import build_data_module, force_nlet_builder
from gorillatracker.model.get_model_cls import get_model_cls
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig
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
warnings.filterwarnings("ignore", ".*No positive samples in targets.*")


def main(args: TrainingArgs) -> None:
    ########### CUDA checks ###########
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
    if args.debug:
        port = 5678
        debugpy.listen(("0.0.0.0", port))
        logger.info(
            f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
        )
        debugpy.wait_for_client()

    args.seed = seed_everything(workers=True, seed=args.seed)

    ############# Construct W&B Logger ##############
    wandb_logging_module = WandbLoggingModule(args)
    wandb_logger = wandb_logging_module.construct_logger()

    ################# Construct model class ##############
    model_cls = get_model_cls(args.model_name_or_path)

    #################### Construct Data Module #################
    def resize_transform(x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize_transform(x: torch.Tensor) -> torch.Tensor:
        return x

    if args.data_resize_transform:
        resize_transform = Resize((args.data_resize_transform, args.data_resize_transform))
    if args.use_normalization:
        normalize_transform = Normalize(eval(args.normalization_mean), eval(args.normalization_std))
    model_transforms = Compose([resize_transform, normalize_transform])

    ssl_config = SSLConfig(
        tff_selection=args.tff_selection,
        negative_mining=args.negative_mining,
        n_samples=args.n_samples,
        feature_types=args.feature_types,
        min_confidence=args.min_confidence,
        min_images_per_tracking=args.min_images_per_tracking,
        width_range=args.width_range,
        height_range=args.height_range,
        split_path=args.split_path,
    )
    if args.force_nlet_builder is not None and args.force_nlet_builder != "None":
        force_nlet_builder(args.force_nlet_builder)

    dm = build_data_module(
        dataset_class_id=args.dataset_class,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        loss_mode=args.loss_mode,
        workers=args.workers,
        model_transforms=model_transforms,
        training_transforms=model_cls.get_training_transforms(),
        additional_eval_datasets_ids=args.additional_val_dataset_classes,
        additional_eval_data_dirs=[Path(d) for d in args.additional_val_data_dirs],
        dataset_names=args.dataset_names,
        ssl_config=ssl_config,
    )

    ################# Construct model ##############
    if not args.kfold:  # NOTE(memben): As we do not yet have the parameters to initalize the model
        model_constructor = ModelConstructor(args, model_cls, dm, wandb_logger)
        model = model_constructor.construct(wandb_logging_module, wandb_logger)

    #################### Trainer #################
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_callback = ModelCheckpoint(
        filename="epoch-{epoch}-acc-{"
        + str(dm.get_dataset_class_names()[0])
        + "/val/embeddings/knn5_crossvideo/accuracy:.3f}",
        monitor=f"{dm.get_dataset_class_names()[0]}/val/embeddings/knn5_crossvideo/accuracy" if not args.use_ssl else f"{dm.get_dataset_class_names()[1]}/val/embeddings/knn5_crossvideo/accuracy",
        mode="max",
        auto_insert_metric_name=False,
        every_n_epochs=int(args.save_interval),
        save_last=True,  # also save the last checkpoint
    )

    early_stopping = EarlyStopping(
        monitor=f"{dm.get_dataset_class_names()[0]}/val/embeddings/knn5_crossvideo/accuracy" if not args.use_ssl else f"{dm.get_dataset_class_names()[1]}/val/embeddings/knn5_crossvideo/accuracy",
        mode="max",
        min_delta=args.min_delta,
        patience=args.early_stopping_patience,
    )

    callbacks = (
        [
            checkpoint_callback,  # keep this at the top
            lr_monitor,
            early_stopping,
        ]
        if not args.kfold
        else [
            lr_monitor,
        ]
    )

    # Initialize trainer
    supported_quantizations = ["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"]
    if args.precision in supported_quantizations:
        args.plugins = BitsandbytesPrecision(mode=args.precision)  # type: ignore
        args.precision = "16-true"

    logger.info(
        f"Total optimizer epochs: {args.max_epochs} | "
        f"Model Log Frequency: {args.save_interval} | "
        f"Effective batch size: {args.batch_size} | "
    )

    ################# Start training #################
    logger.info("Starting training...")
    assert not (
        args.use_quantization_aware_training and args.kfold
    ), "Quantization aware training not supported with kfold"
    if args.kfold:
        train_and_validate_using_kfold(
            args=args,
            dm=dm,
            model_cls=model_cls,
            callbacks=callbacks,
            wandb_logger=wandb_logger,
            wandb_logging_module=wandb_logging_module,
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
    main(parsed_arg_groups)
