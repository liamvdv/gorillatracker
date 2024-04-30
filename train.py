import warnings
from pathlib import Path
from typing import Union

import torch
import wandb
from lightning import Trainer, seed_everything
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

    # Resume from checkpoint if specified
    model_args = dict(
        model_name_or_path=args.model_name_or_path,
        from_scratch=args.from_scratch,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        lr_schedule=args.lr_schedule,
        warmup_mode=args.warmup_mode,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        initial_lr=args.initial_lr,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        stepwise_schedule=args.stepwise_schedule,
        lr_interval=args.val_check_interval,
        margin=args.margin,
        loss_mode=args.loss_mode,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        s=args.s,
        delta_t=args.delta_t,
        mem_bank_start_epoch=args.mem_bank_start_epoch,
        lambda_membank=args.lambda_membank,
        num_classes=(
            (dm.get_num_classes("train"), dm.get_num_classes("val"), dm.get_num_classes("test"))  # type: ignore
            if not args.use_ssl
            else (-1, -1, -1)
        ),
        dropout_p=args.dropout_p,
        accelerator=args.accelerator,
        l2_alpha=args.l2_alpha,
        l2_beta=args.l2_beta,
        path_to_pretrained_weights=args.path_to_pretrained_weights,
        use_wildme_model=args.use_wildme_model,
    )

    if args.saved_checkpoint_path is not None:
        args.saved_checkpoint_path = wandb_logging_module.check_for_wandb_checkpoint_and_download_if_necessary(
            args.saved_checkpoint_path, wandb_logger.experiment
        )

        if args.resume:  # load weights, optimizer states, scheduler state, ...\
            model = model_cls.load_from_checkpoint(args.saved_checkpoint_path, save_hyperparameters=False)
            # we will resume via trainer.fit(ckpt_path=...)
        else:  # load only weights
            model = model_cls(**model_args)  # type: ignore
            # torch_load = torch.load(args.saved_checkpoint_path, map_location=torch.device(args.accelerator))
            # model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = model_cls(**model_args)  # type: ignore

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."
                "Please install torch >= 2.0 or disable compile."
            )
        model = torch.compile(model)

    #################### Construct trainer #################

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
        args.plugins = BitsandbytesPrecision(mode=args.precision)
        args.precision = "bf16-true"

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
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=24,
        # accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=args.fast_dev_run,
        profiler=args.profiler,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
        plugins=args.plugins,
        # reload_dataloaders_every_n_epochs=1,
    )

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

    ########### Start val & train loop ###########
    if args.val_before_training and not args.resume:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        trainer.validate(model, dm)

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

        assert checkpoint_callback.dirpath is not None
        save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
        trainer.save_checkpoint(save_path)

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
