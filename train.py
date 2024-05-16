import warnings
from pathlib import Path
from typing import Union

import torch
import wandb
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.plugins import BitsandbytesPrecision
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import parse
from torchvision.transforms import Compose, Resize

from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank, wait_for_debugger  # type: ignore
from gorillatracker.args import TrainingArgs
from gorillatracker.data_modules import NletDataModule
from gorillatracker.metrics import LogEmbeddingsToWandbCallback
from gorillatracker.model import get_model_cls
from gorillatracker.ssl_pipeline.data_module import SSLDataModule
from gorillatracker.train_utils import get_data_module
from gorillatracker.utils.train import ModelConstructor, train_and_validate_model, train_and_validate_using_kfold
from gorillatracker.utils.wandb_logger import WandbLoggingModule

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*was configured so validation will run at the end of the training epoch.*")
warnings.filterwarnings("ignore", ".*Applied workaround for CuDNN issue.*")


def main(args: TrainingArgs) -> None:  # noqa: C901
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
        dm = SSLDataModule(
            batch_size=args.batch_size,
            transforms=model_transforms,
            training_transforms=model_cls.get_training_transforms(),
            data_dir=str(args.data_dir),
        )
    else:
        dm = get_data_module(
            args.dataset_class,
            str(args.data_dir),
            args.batch_size,
            args.loss_mode,
            model_transforms,
            model_cls.get_training_transforms(),
        )

    ################# Construct model ##############

    model_constructor = ModelConstructor(args, model_cls, dm)
    model = model_constructor.construct(wandb_logging_module, wandb_logger)

    #################### Construct dataloaders & trainer #################
    model_transforms = model.get_tensor_transforms()
    if args.data_resize_transform is not None:
        model_transforms = Compose([Resize(args.data_resize_transform, antialias=True), model_transforms])
    dm = get_data_module(
        args.dataset_class,
        str(args.data_dir),
        args.batch_size,
        args.loss_mode,
        # args.video_data,
        model_transforms,
        model.get_training_transforms(),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    embeddings_logger_callback = LogEmbeddingsToWandbCallback(
        every_n_val_epochs=args.embedding_save_interval,
        knn_with_train=args.knn_with_train,
        wandb_run=wandb_logger.experiment,
        dm=dm,
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
        min_delta=args.min_delta,
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

    if args.pretrained_weights_file is not None:
        # delete everything in model except model.model
        for k in list(model.__dict__.keys()):
            if k != "model" and not k.startswith("_"):
                del model.__dict__[k]
        # trainer.save_checkpoint(str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt"))
        torch.save(model.state_dict(), args.pretrained_weights_file)
        logger.info("Model saved")
        exit(0)

    ### Preperation for quantization aware training ###
    if args.use_quantization_aware_training:
        logger.info("Preperation for quantization aware training...")
        from torch._export import capture_pre_autograd_graph
        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        from gorillatracker.quantization.utils import get_model_input

        example_inputs, _ = get_model_input(dm.dataset_class, str(args.data_dir), amount_of_tensors=100)  # type: ignore
        model.model = capture_pre_autograd_graph(model.model, example_inputs)
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())  # type: ignore
        model.model = prepare_qat_pt2e(model.model, quantizer)  # type: ignore

    ################# Start training #################
    logger.info(f"Rank {current_process_rank} | Starting training...")
    if args.kfold:
        model, trainer = train_and_validate_using_kfold(
            args=args,
            dm=dm,
            model=model,
            callbacks=callbacks,
            wandb_logger=wandb_logger,
            embeddings_logger_callback=embeddings_logger_callback,
        )
    else:
        model, trainer = train_and_validate_model(
            args=args, dm=dm, model=model, callbacks=callbacks, wandb_logger=wandb_logger
        )

    ########### Save checkpoint ###########
    if current_process_rank == 0:
        logger.info("Trying to save checkpoint....")

        assert checkpoint_callback.dirpath is not None
        save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
        trainer.save_checkpoint(save_path)
        logger.info(f"Checkpoint saved to {save_path}")

        if args.save_model_to_wandb:
            logger.info("Collecting PL checkpoint for wandb...")
            artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
            artifact.add_file(save_path, name="model.ckpt")

            logger.info("Pushing to wandb...")
            aliases = ["train_end", "latest"]
            wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

            logger.success("Saving finished!")
    else:
        logger.info("Rank is not 0, skipping checkpoint saving...")


if __name__ == "__main__":
    print("Starting training script...")
    config_path = "./cfgs/config.yml"
    parsed_arg_groups = parse(TrainingArgs, config_path=config_path)

    # parses the config file as default and overwrites with command line arguments
    # therefore allowing sweeps to overwrite the defaults in config file
    current_process_rank = get_rank()
    with graceful_exceptions(extra_message=f"Rank: {current_process_rank}"):
        main(parsed_arg_groups)
