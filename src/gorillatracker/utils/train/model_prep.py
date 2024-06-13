from typing import Any, Type

import torch
from lightning.pytorch.loggers.wandb import WandbLogger

from gorillatracker.args import TrainingArgs
from gorillatracker.data.nlet import NletDataModule
from gorillatracker.model import BaseModule
from gorillatracker.utils.wandb_logger import WandbLoggingModule


class ModelConstructor:
    def __init__(self, args: TrainingArgs, model_cls: Type[BaseModule], dm: NletDataModule) -> None:
        self.args = args
        self.model_cls = model_cls
        self.dm = dm
        self.model_args = self.model_args_from_training_args()

    def model_args_from_training_args(self) -> dict[str, Any]:
        args = self.args

        num_classes = None
        class_distribution = None
        # TODO(memben): this is not logical for multiple datasets
        if "softmax" in args.loss_mode:
            # HACK(memben): To force load the datasets
            self.dm.setup("fit")
            self.dm.setup("test")

            num_classes = (
                self.dm.get_num_classes("train"),
                self.dm.get_num_classes("val"),
                self.dm.get_num_classes("test"),
            )
            class_distribution = (
                self.dm.get_class_distribution("train"),
                self.dm.get_class_distribution("val"),
                self.dm.get_class_distribution("test"),
            )

        dataset_names = self.dm.get_dataset_class_names()

        return dict(
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
            num_classes=num_classes,
            class_distribution=class_distribution,
            dataset_names=dataset_names,
            dropout_p=args.dropout_p,
            accelerator=args.accelerator,
            l2_alpha=args.l2_alpha,
            l2_beta=args.l2_beta,
            path_to_pretrained_weights=args.path_to_pretrained_weights,
            use_wildme_model=args.use_wildme_model,
            k_subcenters=args.k_subcenters,
            use_focal_loss=args.use_focal_loss,
            label_smoothing=args.label_smoothing,
            use_class_weights=args.use_class_weights,
            use_dist_term=args.use_dist_term,
            use_inbatch_mixup=args.use_inbatch_mixup,
            teacher_model_wandb_link=args.teacher_model_wandb_link,
        )

    def construct(
        self,
        wandb_logging_module: WandbLoggingModule,
        wandb_logger: WandbLogger,
    ) -> BaseModule:
        if self.args.saved_checkpoint_path is not None:
            self.args.saved_checkpoint_path = wandb_logging_module.check_for_wandb_checkpoint_and_download_if_necessary(
                self.args.saved_checkpoint_path, wandb_logger.experiment
            )

            if self.args.resume:  # load weights, optimizer states, scheduler state, ...\
                model = self.model_cls.load_from_checkpoint(self.args.saved_checkpoint_path, save_hyperparameters=False)
                # we will resume via trainer.fit(ckpt_path=...)
            else:  # load only weights
                model = self.model_cls(**self.model_args)
                # torch_load = torch.load(args.saved_checkpoint_path, map_location=torch.device(args.accelerator))
                # model.load_state_dict(torch_load["state_dict"], strict=False)
        else:
            model = self.model_cls(**self.model_args)

        if self.args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    f"The current torch version ({torch.__version__}) does not have support for compile."
                    "Please install torch >= 2.0 or disable compile."
                )
            model = torch.compile(model)
        return model
