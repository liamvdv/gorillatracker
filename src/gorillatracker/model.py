import importlib
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import lightning as L
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from print_on_steroids import logger
from torch.optim.adamw import AdamW
from torchvision import transforms
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    efficientnet_v2_l,
    resnet18,
    resnet50,
    resnet152,
)
from transformers import ResNetModel

import gorillatracker.type_helper as gtypes
from gorillatracker.data.nlet import NletDataModule
from gorillatracker.data.utils import flatten_batch, lazy_batch_size
from gorillatracker.losses.get_loss import get_loss
from gorillatracker.metrics import evaluate_embeddings, knn, knn_ssl, log_train_images_to_wandb, tsne
from gorillatracker.model_miewid import GeM, load_miewid_model  # type: ignore
from gorillatracker.utils.labelencoder import LinearSequenceEncoder


def warmup_lr(
    warmup_mode: Literal["linear", "cosine", "exponential", "constant"],
    epoch: int,
    initial_lr: float,
    start_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_mode == "linear":
        return (epoch / warmup_epochs * (start_lr - initial_lr) + initial_lr) / initial_lr
    elif warmup_mode == "cosine":
        return (start_lr - (start_lr - initial_lr) * (np.cos(np.pi * epoch / warmup_epochs) + 1) / 2) / initial_lr
    elif warmup_mode == "exponential":
        decay = (start_lr / initial_lr) ** (1 / warmup_epochs)
        return decay**epoch
    elif warmup_mode == "constant":
        return 1.0
    else:
        raise ValueError(f"Unknown warmup_mode {warmup_mode}")


def linear_lr(epoch: int, n_epochs: int, initial_lr: float, start_lr: float, end_lr: float, **args: Any) -> float:
    return (end_lr + (start_lr - end_lr) * (1 - epoch / n_epochs)) / initial_lr


def cosine_lr(epoch: int, n_epochs: int, initial_lr: float, start_lr: float, end_lr: float, **args: Any) -> float:
    return (end_lr + (start_lr - end_lr) * (np.cos(np.pi * epoch / n_epochs) + 1) / 2) / initial_lr


def exponential_lr(
    epoch: int, n_epochs: float, initial_lr: float, start_lr: float, end_lr: float, **args: Any
) -> float:
    decay = (end_lr / start_lr) ** (1 / n_epochs)
    return start_lr * (decay**epoch) / initial_lr


def schedule_lr(
    lr_schedule_mode: Literal["linear", "cosine", "exponential", "constant"],
    epochs: int,
    initial_lr: float,
    start_lr: float,
    end_lr: float,
    n_epochs: int,
) -> float:
    if lr_schedule_mode == "linear":
        return linear_lr(epochs, n_epochs, initial_lr, start_lr, end_lr)
    elif lr_schedule_mode == "cosine":
        return cosine_lr(epochs, n_epochs, initial_lr, start_lr, end_lr)
    elif lr_schedule_mode == "exponential":
        return exponential_lr(epochs, n_epochs, initial_lr, start_lr, end_lr)
    elif lr_schedule_mode == "constant":
        return 1.0
    else:
        raise ValueError(f"Unknown lr_schedule_mode {lr_schedule_mode}")


def combine_schedulers(
    warmup_mode: Literal["linear", "cosine", "exponential", "constant"],
    lr_schedule_mode: Literal["linear", "cosine", "exponential", "constant"],
    epochs: int,
    initial_lr: float,
    start_lr: float,
    end_lr: float,
    n_epochs: int,
    warmup_epochs: int,
) -> float:
    if epochs < warmup_epochs:  # 0 : warmup_epochs - 1
        return warmup_lr(warmup_mode, epochs, initial_lr, start_lr, warmup_epochs)
    else:  # warmup_epochs - 1 : n_epochs - 1
        return schedule_lr(
            lr_schedule_mode, epochs - warmup_epochs, initial_lr, start_lr, end_lr, n_epochs - warmup_epochs
        )


def in_batch_mixup(data: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixes the data and targets in a batch randomly. Targets need to be one-hot encoded."""
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(data.device)  # f(x; a,b) = const * x^(a-1) * (1-x)^(b-1)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


Runner = Any


class BaseModule(L.LightningModule):
    """
    must be subclassed and set self.model = ...
    """

    def __init__(
        self,
        data_module: NletDataModule,
        from_scratch: bool,
        wandb_run: Runner,
        loss_mode: str,
        weight_decay: float,
        lr_schedule: Literal["linear", "cosine", "exponential", "constant", "reduce_on_plateau"],
        warmup_mode: Literal["linear", "cosine", "exponential", "constant"],
        warmup_epochs: int,
        max_epochs: int,
        initial_lr: float,
        start_lr: float,
        end_lr: float,
        stepwise_schedule: bool,
        lr_interval: int,
        beta1: float,
        beta2: float,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        embedding_size: int = 256,
        batch_size: int = 32,
        dataset_names: list[str] = [],
        accelerator: str = "cpu",
        dropout_p: float = 0.0,
        use_dist_term: bool = False,
        use_inbatch_mixup: bool = False,
        kfold_k: Optional[int] = None,
        knn_with_train: bool = False,
        use_quantization_aware_training: bool = False,
        every_n_val_epochs: int = 1,
        fast_dev_run: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        ####### Optimizer and Scheduler
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.warmup_mode = warmup_mode
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.stepwise_schedule = stepwise_schedule
        self.lr_interval = lr_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        ####### Embedding Layers
        self.from_scratch = from_scratch
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p

        ####### Losses
        assert "softmax" in loss_mode or not use_inbatch_mixup, "In-batch mixup is only supported for softmax losses."

        self.loss_mode = loss_mode
        self.use_dist_term = use_dist_term
        self.use_inbatch_mixup = use_inbatch_mixup

        ####### Other
        self.quant = torch.quantization.QuantStub()  # type: ignore
        self.kfold_k = kfold_k
        self.use_quantization_aware_training = use_quantization_aware_training
        self.knn_with_train = knn_with_train
        self.wandb_run = wandb_run
        self.fast_dev_run = fast_dev_run
        self.every_n_val_epochs = every_n_val_epochs
        self.dm = data_module

        ##### Create List of embeddings_tables
        self.embeddings_table_columns = [
            "label",
            "embedding",
            "id",
            "partition",  # val for validation, train for training
            "dataset",
        ]  # note that the dataloader usually returns the order (id, embedding, label)
        self.dataset_names = dataset_names
        self.embeddings_table_list = [
            pd.DataFrame(columns=self.embeddings_table_columns) for _ in range(len(self.dataset_names))
        ]

    def set_losses(
        self,
        model: nn.Module,
        loss_mode: str,
        margin: float,
        s: float,
        delta_t: int,
        mem_bank_start_epoch: int,
        lambda_membank: float,
        embedding_size: int,
        batch_size: int,
        num_classes: Optional[tuple[int, int, int]],
        class_distribution: Optional[dict[int, int]],
        use_focal_loss: bool,
        k_subcenters: int,
        accelerator: str,
        label_smoothing: float,
        l2_alpha: float,
        l2_beta: float,
        path_to_pretrained_weights: str,
        use_class_weights: bool,
        use_dist_term: bool,
        **kwargs: Any,
    ) -> None:
        kfold_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""
        self.loss_module_train = get_loss(
            loss_mode,
            margin=margin,
            embedding_size=embedding_size,
            batch_size=batch_size,
            delta_t=delta_t,
            s=s,
            num_classes=num_classes[0] if num_classes is not None else None,  # TODO(memben)
            class_distribution=class_distribution[0] if class_distribution is not None else None,  # TODO(memben)
            mem_bank_start_epoch=mem_bank_start_epoch,
            lambda_membank=lambda_membank,
            accelerator=accelerator,
            l2_alpha=l2_alpha,
            l2_beta=l2_beta,
            path_to_pretrained_weights=path_to_pretrained_weights,
            use_focal_loss=use_focal_loss,
            label_smoothing=label_smoothing,
            model=model,
            k_subcenters=k_subcenters,
            use_class_weights=use_class_weights,
            use_dist_term=use_dist_term,
            # log_func=lambda x, y: self.log("train/"+ x, y, on_epoch=True),
            log_func=lambda x, y: self.log(f"{kfold_prefix}{x}", y),
            teacher_model_wandb_link=kwargs.get("teacher_model_wandb_link", ""),
        )
        self.loss_module_val = get_loss(
            loss_mode,
            margin=margin,
            embedding_size=embedding_size,
            batch_size=batch_size,
            delta_t=delta_t,
            s=s,
            num_classes=num_classes[1] if num_classes is not None else None,  # TODO(memben)
            class_distribution=class_distribution[1] if class_distribution is not None else None,  # TODO(memben)
            mem_bank_start_epoch=mem_bank_start_epoch,
            lambda_membank=lambda_membank,
            accelerator=accelerator,
            l2_alpha=l2_alpha,
            l2_beta=l2_beta,
            path_to_pretrained_weights=path_to_pretrained_weights,
            use_focal_loss=use_focal_loss,
            label_smoothing=label_smoothing,
            model=model,
            log_func=lambda x, y: self.log(f"{kfold_prefix}{x}", y),
            k_subcenters=1,
            use_class_weights=use_class_weights,
            use_dist_term=use_dist_term,
            teacher_model_wandb_link=kwargs.get("teacher_model_wandb_link", ""),
        )
        self.loss_module_val.eval()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        log_train_images_to_wandb(self.wandb_run, self.trainer, self.dm.train_dataloader(), n_samples=1)

        if self.loss_mode.endswith("vpl") and self.trainer.current_epoch >= self.loss_module_train.mem_bank_start_epoch:  # type: ignore
            self.loss_module_train.set_using_memory_bank(True)  # type: ignore
            logger.info("Using memory bank")
        elif (
            self.loss_mode.endswith("vpl/l2sp")
            and self.trainer.current_epoch >= self.loss_module_train.loss.mem_bank_start_epoch  # type: ignore
        ):  # is wrapped in l2sp regularization
            self.loss_module_train.loss.set_using_memory_bank(True)  # type: ignore
            logger.info("Using memory bank")

    def perform_mixup(self, flat_images: torch.Tensor, flat_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_classes: int
        if "l2sp" in self.loss_mode and not self.use_dist_term:
            num_classes = self.loss_module_train.loss.num_classes  # type: ignore
            flat_labels = self.loss_module_train.loss.le.encode_list(flat_labels.tolist())  # type: ignore
        elif "l2sp" in self.loss_mode and self.use_dist_term:
            num_classes = self.loss_module_train.loss.arcface.num_classes  # type: ignore
            flat_labels = self.loss_module_train.loss.arcface.le.encode_list(flat_labels.tolist())  # type: ignore
        elif "l2sp" not in self.loss_mode and self.use_dist_term:
            num_classes = self.loss_module_train.arcface.num_classes  # type: ignore
            flat_labels = self.loss_module_train.arcface.le.encode_list(flat_labels.tolist())  # type: ignore
        else:
            num_classes = self.loss_module_train.num_classes  # type: ignore
            flat_labels = self.loss_module_train.le.encode_list(flat_labels.tolist())  # type: ignore

        flat_labels = torch.tensor(flat_labels).to(flat_images.device)
        flat_labels_onehot = torch.nn.functional.one_hot(flat_labels, num_classes).float()
        flat_images, flat_labels_onehot = in_batch_mixup(flat_images, flat_labels_onehot)

        return flat_images, flat_labels_onehot

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        _, images, _ = batch
        _, flat_images, flat_labels = flatten_batch(batch)

        flat_labels_onehot = None
        if self.use_inbatch_mixup:
            flat_images, flat_labels_onehot = self.perform_mixup(flat_images, flat_labels)
        embeddings = self.forward(flat_images)

        assert not torch.isnan(embeddings).any(), f"Embeddings are NaN: {embeddings}"
        loss, pos_dist, neg_dist = self.loss_module_train(embeddings=embeddings, labels=flat_labels, images=images, labels_onehot=flat_labels_onehot)  # type: ignore

        log_str_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""
        self.log(f"{log_str_prefix}train/negative_distance", neg_dist, on_step=True)
        self.log(f"{log_str_prefix}train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log(f"{log_str_prefix}train/positive_distance", pos_dist, on_step=True)
        self.log(f"{log_str_prefix}train/negative_distance", neg_dist, on_step=True)
        return loss

    def add_validation_embeddings(
        self,
        anchor_ids: list[str],
        anchor_embeddings: torch.Tensor,
        anchor_labels: torch.Tensor,
        dataloader_idx: int,
    ) -> None:
        # save anchor embeddings of validation step for later analysis in W&B
        embeddings = torch.reshape(anchor_embeddings, (-1, self.embedding_size))
        embeddings = embeddings.cpu()

        assert len(self.embeddings_table_columns) == 5, "Embeddings table columns are not set correctly."
        anchor_labels_list = anchor_labels.tolist()
        data = {
            self.embeddings_table_columns[0]: [int(x) for x in anchor_labels_list],
            self.embeddings_table_columns[1]: [embedding.numpy() for embedding in embeddings],
            self.embeddings_table_columns[2]: anchor_ids,
            self.embeddings_table_columns[3]: "val",
            self.embeddings_table_columns[4]: self.dataset_names[dataloader_idx],
        }

        df = pd.DataFrame(data)
        self.embeddings_table_list[dataloader_idx] = pd.concat(
            [df, self.embeddings_table_list[dataloader_idx]], ignore_index=True
        )
        # NOTE(rob2u): will get flushed by W&B Callback on val epoch end.

    def validation_step(self, batch: gtypes.NletBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        dataloader_name = self.dataset_names[dataloader_idx]

        batch_size = lazy_batch_size(batch)
        _, images, _ = batch
        flat_ids, flat_images, flat_labels = flatten_batch(batch)
        anchor_ids = list(flat_ids[:batch_size])

        embeddings = self.forward(flat_images)

        self.add_validation_embeddings(anchor_ids, embeddings[:batch_size], flat_labels[:batch_size], dataloader_idx)
        if "softmax" not in self.loss_mode and not self.use_dist_term:
            loss, pos_dist, neg_dist = self.loss_module_val(embeddings=embeddings, labels=flat_labels, images=images)  # type: ignore
            kfold_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""
            self.log(
                f"{dataloader_name}/{kfold_prefix}val/loss",
                loss,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{dataloader_name}/{kfold_prefix}val/positive_distance",
                pos_dist,
                on_step=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{dataloader_name}/{kfold_prefix}val/negative_distance",
                neg_dist,
                on_step=True,
                add_dataloader_idx=False,
            )
            return loss
        else:
            return torch.tensor(0.0)  # TODO(memben): ???

    def on_validation_epoch_end(self, dataloader_idx: int = 0) -> None:
        # TODO(rob2u): this fails, IDK why gradient calc is activated, params are not frozen
        # if self.trainer.model.dtype == torch.float32 and not self.use_quantization_aware_training:  # type: ignore
        #     log_grad_cam_images_to_wandb(self.wandb_run, self.trainer, self.dm.train_dataloader())

        dataloader_name = self.dataset_names[dataloader_idx]
        kfold_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""

        if "softmax" in self.loss_mode:
            self.validation_loss_softmax(dataloader_name, kfold_prefix)

        embeddings_table_list = self.embeddings_table_list

        assert self.trainer.max_epochs is not None
        for dataloader_idx, embeddings_table in enumerate(embeddings_table_list):
            self.eval_embeddings_table(embeddings_table, dataloader_idx)

        # clear the table where the embeddings are stored
        self.embeddings_table_list = [
            pd.DataFrame(columns=self.embeddings_table_columns) for _ in range(len(self.dataset_names))
        ]  # reset embeddings table

    def lambda_schedule(self, epoch: int) -> float:
        return combine_schedulers(
            self.warmup_mode,
            self.lr_schedule,  # type: ignore
            epoch,
            self.initial_lr,
            self.start_lr,
            self.end_lr,
            self.max_epochs,
            self.warmup_epochs,
        )

    def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRSchedulerConfig:
        logger.info(
            f"Using {self.lr_schedule} learning rate schedule with {self.warmup_mode} warmup for {self.max_epochs} epochs."
        )
        if "l2sp" in self.loss_mode and self.weight_decay != 0.0:
            logger.warning(
                "Using L2SP regularization, weight decay will be set to 0.0. Please use the l2_alpha and l2_beta arguments to set the L2SP parameters."
            )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay if "l2sp" not in self.loss_mode else 0.0,
        )
        if self.lr_schedule != "reduce_on_plateau":
            lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=self.lambda_schedule,
            )
            if self.stepwise_schedule:
                lr_scheduler: LRSchedulerConfigType = {
                    "scheduler": lambda_scheduler,
                    "interval": "step",
                    "frequency": self.lr_interval,
                }
            else:
                lr_scheduler = {"scheduler": lambda_scheduler, "interval": "epoch"}

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }
        else:
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=self.lr_decay,
                patience=self.lr_decay_interval,
                verbose=True,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            )
            return {"optimizer": optimizer, "lr_scheduler": plateau_scheduler}

    def _get_train_embeddings_for_knn(self, trainer: L.Trainer) -> Tuple[torch.Tensor, torch.Tensor, list[gtypes.Id]]:
        assert trainer.model is not None, "Model must be initalized before validation phase."
        train_embedding_batches = []
        train_labels = torch.tensor([])
        train_ids: list[gtypes.Id] = []
        for batch in self.dm.train_dataloader():
            batch_size = lazy_batch_size(batch)
            flat_ids, flat_images, flat_labels = flatten_batch(batch)
            anchor_labels = flat_labels[:batch_size]
            anchor_images = flat_images[:batch_size].to(trainer.model.device)
            anchor_ids = flat_ids[:batch_size]
            embeddings = trainer.model(anchor_images)
            train_embedding_batches.append(embeddings)
            train_labels = torch.cat([train_labels, anchor_labels], dim=0)
            train_ids.extend(anchor_ids)
        train_embeddings = torch.cat(train_embedding_batches, dim=0)
        assert len(train_embeddings) == len(train_labels)
        return train_embeddings.cpu(), train_labels.cpu(), train_ids

    def eval_embeddings_table(self, embeddings_table: pd.DataFrame, dataloader_idx: int) -> None:
        dataloader_name = self.dm.get_dataset_class_names()[dataloader_idx]
        if self.knn_with_train:
            train_embeddings, train_labels, train_ids = self._get_train_embeddings_for_knn(self.trainer)

            # add train embeddings to the embeddings table
            embeddings_table = pd.concat(
                [
                    embeddings_table,
                    pd.DataFrame(
                        {
                            "label": [int(x) for x in train_labels.tolist()],
                            "embedding": train_embeddings.tolist(),
                            "id": train_ids,
                            "partition": "train",
                            "dataset": dataloader_name,
                        }
                    ),
                ],
                ignore_index=True,
            )

        metrics = {
            "knn5": partial(knn, k=5),
            "knn": partial(knn, k=1),
            "knn5_macro": partial(knn, k=5, average="macro"),
            "knn_macro": partial(knn, k=1, average="macro"),
            "tsne": tsne,
            # "pca": pca,
            # "fc_layer": fc_layer,
        }
        metrics |= (
            {
                "knn5-with-train": partial(knn, k=5, use_train_embeddings=True),
                "knn-with-train": partial(knn, k=1, use_train_embeddings=True),
                "knn5-with-train_macro": partial(knn, k=5, use_train_embeddings=True, average="macro"),
                "knn-with-train_macro": partial(knn, k=1, use_train_embeddings=True, average="macro"),
            }
            if self.knn_with_train
            else {}
        )
        metrics |= (
            {
                "knn_crossencounter": partial(knn, k=1, use_crossvideo_positives=True),
                "knn5_crossencounter": partial(knn, k=5, use_crossvideo_positives=True),
                "knn_crossencounter_macro": partial(knn, k=1, use_crossvideo_positives=True, average="macro"),
                "knn5_crossencounter_macro": partial(knn, k=5, use_crossvideo_positives=True, average="macro"),
            }
            if "CXL" in dataloader_name or "Bristol" in dataloader_name
            else {}
        )
        metrics = (
            {
                "knn_ssl": partial(knn_ssl, k=1, dm=self.dm),
                "knn5_ssl": partial(knn_ssl, k=5, dm=self.dm),
                "knn_ssl_macro": partial(knn_ssl, k=1, dm=self.dm, average="macro"),
                "knn5_ssl_macro": partial(knn_ssl, k=5, dm=self.dm, average="macro"),
            }
            if "ssl" in dataloader_name.lower()
            else metrics
        )

        metrics = metrics if not self.fast_dev_run else {}

        # log to wandb
        evaluate_embeddings(
            data=embeddings_table,
            embedding_name="val/embeddings",
            metrics=metrics,
            kfold_k=self.kfold_k if hasattr(self, "kfold_k") else None,  # TODO(memben)
            dataloader_name=dataloader_name,
        )

    def validation_loss_softmax(self, dataloader_name: str, kfold_prefix: str) -> None:
        for i, table in enumerate(self.embeddings_table_list):
            logger.info(f"Calculating loss for all embeddings from dataloader {i}: {len(table)}")

            # get weights for all classes by averaging over all embeddings
            loss_module_val = (
                self.loss_module_val if not self.loss_mode.endswith("l2sp") else self.loss_module_val.loss  # type: ignore
            )
            if self.use_dist_term:
                loss_module_val = loss_module_val.arcface

            num_classes = table["label"].nunique()  # TODO(memben + rob2u)
            assert len(table) > 0, f"Empty table for dataloader {i}"

            # get weights for all classes by averaging over all embeddings
            class_weights = torch.zeros(num_classes, self.embedding_size).to(self.device)
            lse = LinearSequenceEncoder()
            table["label"] = table["label"].apply(lse.encode)

            for label in range(num_classes):
                class_embeddings = table[table["label"] == label]["embedding"].tolist()
                class_embeddings = (
                    np.stack(class_embeddings) if len(class_embeddings) > 0 else np.zeros((0, self.embedding_size))
                )
                class_weights[label] = torch.tensor(class_embeddings).mean(dim=0)
                if torch.isnan(class_weights[label]).any():
                    class_weights[label] = 0.0

            # calculate loss for all embeddings
            loss_module_val.update(class_weights, num_classes, lse)

            losses = []
            for _, row in table.iterrows():
                loss, _, _ = loss_module_val(
                    torch.tensor(row["embedding"]).unsqueeze(0),
                    torch.tensor(lse.decode(row["label"])).unsqueeze(0),
                )
                losses.append(loss)
            loss = torch.tensor(losses).mean()
            assert not torch.isnan(loss).any(), f"Loss is NaN: {losses}"
            self.log(f"{dataloader_name}/{kfold_prefix}val/loss", loss, sync_dist=True)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        """Add your data augmentations here. Function will be called after in the training loop"""
        return lambda x: x

    @classmethod
    def get_tensor_transforms(cls) -> None:
        raise NotImplementedError(
            "This method was deprecated! Use arg.use_normalization and args.data_resize_transform instead!"
        )


class EfficientNetV2Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        is_from_scratch = kwargs.get("from_scratch", False)
        self.model = (
            efficientnet_v2_l()
            if is_from_scratch
            else efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        )
        # self.model.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.embedding_size),
        # )
        self.model.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.classifier[1].in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # return self.model.blocks[-1].conv
        return self.model.features[-1][0]  # TODO(liamvdv)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


class EfficientNetRW_M(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        is_from_scratch = kwargs.get("from_scratch", False)
        self.model = timm.create_model("efficientnetv2_rw_m", pretrained=not is_from_scratch)

        self.model.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.classifier.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.classifier.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        return self.model.conv_head

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(192, scale=(0.75, 1.0)),
                # transforms_v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                # transforms_v2.RandomPerspective(distortion_scale=0.8, p=1.0, fill=0),
            ]
        )


class ConvNeXtV2BaseWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("convnextv2_base", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        return self.model.stages[-1].blocks[-1].conv_dw

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, scale=(0.02, 0.13)),
            ]
        )


class ConvNeXtV2HugeWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("convnextv2_huge", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)


class VisionTransformerWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_large_patch16_224", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-vision-transformers
        return self.model.blocks[-1].norm1

    def get_grad_cam_reshape_transform(self) -> Any:
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-vision-transformers
        def reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

            result = result.transpose(2, 3).transpose(1, 2)
            return result

        return reshape_transform

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


class VisionTransformerDinoV2Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch)
        self.model.reset_classifier(self.embedding_size)
        self.model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class VisionTransformerClipWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_base_patch16_clip_224.metaclip_2pt5b", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ConvNextClipWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        model_name = "convnext_base.clip_laion2b"
        self.model = (
            timm.create_model(model_name, pretrained=False)
            if kwargs.get("from_scratch", False)
            else timm.create_model(model_name, pretrained=True)
        )
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ConvNextWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("convnext_base", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class SwinV2BaseWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        swin_model = "swinv2_base_window12_192.ms_in22k"
        self.model = (
            timm.create_model(swin_model, pretrained=False)
            if kwargs.get("from_scratch", False)
            else timm.create_model(swin_model, pretrained=True)
        )
        # self.model.head.fc = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
        # ) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-swin-transformers
        return self.model.layers[-1].blocks[-1].norm1

    def get_grad_cam_reshape_transform(self) -> Any:
        # Implementation for "swin_base_patch4_window7_224"
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-swin-transformers

        # NOTE(liamvdv): we use this implementation for "swinv2_base_window12_192.ms_in22k"
        # TODO(liamvdv): I'm not sure this is correct, but it seems to work...
        def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
            batch_size, _, _, _ = tensor.shape
            total_elements = tensor.numel()
            num_channels = total_elements // (batch_size * 12 * 12)

            result = tensor.reshape(batch_size, num_channels, 12, 12)
            return result

        return reshape_transform

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(192, scale=(0.75, 1.0)),
            ]
        )


class SwinV2LargeWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        swin_model = "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
        self.model = (
            timm.create_model(swin_model, pretrained=False)
            if kwargs.get("from_scratch", False)
            else timm.create_model(swin_model, pretrained=True)
        )
        # self.model.head.fc = torch.nn.Linear(
        #     in_features=self.model.head.fc.in_features, out_features=self.embedding_size
        # ) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


class ResNet18Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = (
            resnet18() if kwargs.get("from_scratch", False) else resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        )
        # self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size) # TODO
        self.model.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # return self.model.layer4[-1]
        return self.model.layer4[-1].conv2

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ResNet152Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = (
            resnet152() if kwargs.get("from_scratch", False) else resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        )
        # self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size) # TODO
        self.model.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # return self.model.layer4[-1]
        return self.model.layer4[-1].conv3

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ResNet50Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = (
            resnet50() if kwargs.get("from_scratch", False) else resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        )
        # self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size) # TODO
        self.model.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ResNet50DinoV2Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")
        # self.last_linear = torch.nn.Linear(in_features=2048, out_features=self.embedding_size) # TODO
        self.last_linear = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=2048, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        feature_vector = gap(outputs.last_hidden_state)
        feature_vector = torch.flatten(feature_vector, start_dim=2).squeeze(-1)
        feature_vector = self.last_linear(feature_vector)
        return feature_vector

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class InceptionV3Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("inception_v3", pretrained=not self.from_scratch)

        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        return self.model.Mixed_7c.branch_pool

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


class MiewIdNetWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        is_from_scratch = kwargs.get("from_scratch", False)
        use_wildme_model = kwargs.get("use_wildme_model", False)

        if use_wildme_model:
            logger.info("Using WildMe model")
            self.model = load_miewid_model()
            # fix model
            for param in self.model.parameters():
                param.requires_grad = False

            # self.model.global_pool = nn.Identity()
            # self.model.bn = nn.Identity()
            self.classifier = torch.nn.Sequential(
                # torch.nn.BatchNorm1d(2152),
                torch.nn.Dropout(p=self.dropout_p),
                torch.nn.Linear(in_features=2152, out_features=self.embedding_size),
                torch.nn.BatchNorm1d(self.embedding_size),
            )
            self.set_losses(self.model, **kwargs)
            return

        self.model = timm.create_model("efficientnetv2_rw_m", pretrained=not is_from_scratch)
        in_features = self.model.classifier.in_features

        self.model.global_pool = nn.Identity()  # NOTE: GeM = Generalized Mean Pooling
        self.model.classifier = nn.Identity()

        # TODO(rob2u): load wildme model weights here then initialize the classifier and get loss modes -> change the transforms accordingly (normalize, etc.)
        self.classifier = torch.nn.Sequential(
            GeM(),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        return self.model.blocks[-1][-1].conv_pwl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.classifier(x)
        return x

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(440, scale=(0.75, 1.0)),
            ]
        )


# NOTE(liamvdv): Register custom model backbones here.
custom_model_cls = {
    "EfficientNetV2_Large": EfficientNetV2Wrapper,
    "SwinV2Base": SwinV2BaseWrapper,
    "SwinV2LargeWrapper": SwinV2LargeWrapper,
    "ViT_Large": VisionTransformerWrapper,
    "ResNet18": ResNet18Wrapper,
    "ResNet152": ResNet152Wrapper,
    "ResNet50Wrapper": ResNet50Wrapper,
    "ResNet50DinoV2Wrapper": ResNet50DinoV2Wrapper,
    "ConvNeXtV2_Base": ConvNeXtV2BaseWrapper,
    "ConvNeXtV2_Huge": ConvNeXtV2HugeWrapper,
    "ConvNextWrapper": ConvNextWrapper,
    "ConvNextClipWrapper": ConvNextClipWrapper,
    "VisionTransformerDinoV2": VisionTransformerDinoV2Wrapper,
    "VisionTransformerClip": VisionTransformerClipWrapper,
    "MiewIdNet": MiewIdNetWrapper,
    "EfficientNet_RW_M": EfficientNetRW_M,
    "InceptionV3": InceptionV3Wrapper,
}


def get_model_cls(model_name: str) -> Type[BaseModule]:
    model_cls = custom_model_cls.get(model_name, None)
    if not model_cls:
        module, cls = model_name.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module), cls)
    return model_cls
