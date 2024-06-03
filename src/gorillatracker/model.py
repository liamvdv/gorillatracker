import importlib
from itertools import chain
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

import lightning as L
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from facenet_pytorch import InceptionResnetV1
from print_on_steroids import logger
from torch.optim import AdamW
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
from gorillatracker.losses.arcface_loss import VariationalPrototypeLearning
from gorillatracker.losses.triplet_loss import L2SPRegularization_Wrapper, get_loss
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


class BaseModule(L.LightningModule):
    """
    must be subclassed and set self.model = ...
    """

    def __init__(
        self,
        model_name_or_path: str,
        # model_kwargs: dict,
        from_scratch: bool,
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
        margin: float = 0.5,
        s: float = 64.0,
        delta_t: int = 200,
        mem_bank_start_epoch: int = 2,
        lambda_membank: float = 0.5,
        embedding_size: int = 256,
        batch_size: int = 32,
        num_classes: Tuple[int, int, int] = (0, 0, 0),
        accelerator: str = "cpu",
        dropout_p: float = 0.0,
        num_val_dataloaders: int = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

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
        self.margin = margin

        self.from_scratch = from_scratch
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.loss_mode = loss_mode

        self.quant = torch.quantization.QuantStub()  # type: ignore

        ##### Create List of embeddings_tables
        self.embeddings_table_columns = [
            "label",
            "embedding",
            "id",
        ]  # note that the dataloader usually returns the order (id, embedding, label)
        self.num_val_dataloaders = num_val_dataloaders
        self.embeddings_table_list = [
            pd.DataFrame(columns=self.embeddings_table_columns) for _ in range(self.num_val_dataloaders)
        ]

    def set_losses(
        self,
        model: nn.Module,
        loss_mode: str,
        s: float = 64.0,
        delta_t: int = 200,
        mem_bank_start_epoch: int = 2,
        lambda_membank: float = 0.5,
        embedding_size: int = 256,
        batch_size: int = 32,
        num_classes: Tuple[int, int, int] = (0, 0, 0),
        accelerator: str = "cpu",
        **kwargs: Dict[str, Any],
    ) -> None:
        self.loss_module_train = get_loss(
            loss_mode,
            margin=self.margin,
            embedding_size=self.embedding_size,
            batch_size=batch_size,
            delta_t=delta_t,
            s=s,
            num_classes=num_classes[0],
            mem_bank_start_epoch=mem_bank_start_epoch,
            lambda_membank=lambda_membank,
            accelerator=accelerator,
            l2_alpha=kwargs["l2_alpha"],
            l2_beta=kwargs["l2_beta"],
            path_to_pretrained_weights=kwargs["path_to_pretrained_weights"],
            model=model,
            log_func=self.log,
        )
        self.loss_module_val = get_loss(
            loss_mode,
            margin=self.margin,
            embedding_size=self.embedding_size,
            batch_size=batch_size,
            delta_t=delta_t,
            s=s,
            num_classes=num_classes[1],
            mem_bank_start_epoch=mem_bank_start_epoch,
            lambda_membank=lambda_membank,
            accelerator=accelerator,
            l2_alpha=kwargs["l2_alpha"],
            l2_beta=kwargs["l2_beta"],
            path_to_pretrained_weights=kwargs["path_to_pretrained_weights"],
            model=model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        if (
            isinstance(self.loss_module_train, VariationalPrototypeLearning)
            and self.trainer.current_epoch >= self.loss_module_train.mem_bank_start_epoch
        ):
            self.loss_module_train.set_using_memory_bank(True)
            logger.info("Using memory bank")
        elif (
            isinstance(self.loss_module_train, L2SPRegularization_Wrapper)
            and isinstance(self.loss_module_train.loss, VariationalPrototypeLearning)
            and self.trainer.current_epoch >= self.loss_module_train.loss.mem_bank_start_epoch
        ):  # is wrapped in l2sp regularization
            self.loss_module_train.loss.set_using_memory_bank(True)
            logger.info("Using memory bank")

    # TODO(memben): ATTENTION: type hints NOT correct, only for SSL
    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        ids, images, labels = batch

        # HACK(memben): We'll allow this for now, but we should correct it later
        if torch.is_tensor(labels[0]):  # type: ignore
            flat_labels = torch.cat(labels, dim=0)  # type: ignore
            vec = torch.cat(images, dim=0)  # type: ignore
        else:
            # NOTE(memben): this is the expected shape
            # transform ((a1, p1, n1), (a2, p2, n2)) to (a1, a2, p1, p2, n1, n2)
            flat_labels = torch.cat([torch.Tensor(d) for d in zip(*labels)], dim=0)
            # transform ((a1: Tensor, p1: Tensor, n1: Tensor), (a2, p2, n2)) to (a1, a2, p1, p2, n1, n2)
            vec = torch.stack(list(chain.from_iterable(zip(*images))), dim=0)
        embeddings = self.forward(vec)

        loss, pos_dist, neg_dist = self.loss_module_train(embeddings, flat_labels)  # type: ignore
        self.log("train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/positive_distance", pos_dist, on_step=True)
        self.log("train/negative_distance", neg_dist, on_step=True)
        return loss

    def add_validation_embeddings(
        self,
        anchor_ids: List[str],
        anchor_embeddings: torch.Tensor,
        anchor_labels: gtypes.MergedLabels,
        dataloader_idx: int,
    ) -> None:
        # save anchor embeddings of validation step for later analysis in W&B
        embeddings = torch.reshape(anchor_embeddings, (-1, self.embedding_size))
        embeddings = embeddings.cpu()

        assert len(self.embeddings_table_columns) == 3
        data = {
            self.embeddings_table_columns[0]: (anchor_labels.tolist()),  # type: ignore
            self.embeddings_table_columns[1]: [embedding.numpy() for embedding in embeddings],
            self.embeddings_table_columns[2]: anchor_ids,
        }

        df = pd.DataFrame(data)
        self.embeddings_table_list[dataloader_idx] = pd.concat(
            [df, self.embeddings_table_list[dataloader_idx]], ignore_index=True
        )
        # NOTE(rob2u): will get flushed by W&B Callback on val epoch end.

    # TODO(memben): ATTENTION: type hints NOT correct, only for SSL
    def validation_step(self, batch: gtypes.NletBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        ids, images, labels = batch  # embeddings either (ap, a, an, n) or (a, p, n)
        n_anchors = len(images[0])

        # HACK(memben): We'll allow this for now, but we should correct it later
        if torch.is_tensor(labels[0]):  # type: ignore
            flat_labels = torch.cat(labels, dim=0)  # type: ignore
            vec = torch.cat(images, dim=0)  # type: ignore
        else:
            # NOTE(memben): this is the expected shape
            flat_labels = torch.cat([torch.Tensor(d) for d in zip(*labels)], dim=0)
            vec = torch.stack(list(chain.from_iterable(zip(*images))), dim=0)

        flat_ids = [id for nlet in ids for id in nlet]  # TODO(memben): This seems to be wrong for SSL
        embeddings = self.forward(vec)
        self.add_validation_embeddings(
            flat_ids[:n_anchors], embeddings[:n_anchors], flat_labels[:n_anchors], dataloader_idx
        )
        if "softmax" not in self.loss_mode:
            loss, pos_dist, neg_dist = self.loss_module_val(embeddings, flat_labels)  # type: ignore
            self.log(
                f"val/loss/dataloader_{dataloader_idx}",
                loss,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/positive_distance/dataloader_{dataloader_idx}", pos_dist, on_step=True, add_dataloader_idx=False
            )
            self.log(
                f"val/negative_distance/dataloader_{dataloader_idx}", neg_dist, on_step=True, add_dataloader_idx=False
            )
            return loss
        else:
            return torch.tensor(0.0)

    def on_validation_epoch_end(self) -> None:
        # calculate loss after all embeddings have been processed
        if "softmax" in self.loss_mode:
            for i, table in enumerate(self.embeddings_table_list):
                logger.info(f"Calculating loss for all embeddings from dataloader {i}: {len(table)}")

            # get weights for all classes by averaging over all embeddings
            loss_module_val = (
                self.loss_module_val
                if not isinstance(self.loss_module_val, L2SPRegularization_Wrapper)
                else self.loss_module_val.loss  # type: ignore
            )
            num_classes = (
                self.loss_module_val.num_classes  # type: ignore
                if not isinstance(self.loss_module_val, L2SPRegularization_Wrapper)
                else self.loss_module_val.loss.num_classes  # type: ignore
            )

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
            loss_module_val.set_weights(class_weights)  # type: ignore
            loss_module_val.le = lse  # type: ignore

            losses = []
            for _, row in table.iterrows():
                loss, _, _ = loss_module_val(
                    torch.tensor(row["embedding"]).unsqueeze(0),
                    torch.tensor(lse.decode(row["label"])).unsqueeze(0),  # type: ignore
                )
                losses.append(loss)
            loss = torch.tensor(losses).mean()
            assert not torch.isnan(loss).any(), f"Loss is NaN: {losses}"
            self.log(f"val/loss/dataloader_{i}", loss, sync_dist=True)

        # clear the table where the embeddings are stored
        self.embeddings_table_list = [
            pd.DataFrame(columns=self.embeddings_table_columns) for _ in range(self.num_val_dataloaders)
        ]  # reset embeddings table

    def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRSchedulerConfig:
        if self.global_rank == 0:
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

        # optimizer = torch.optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.initial_lr,
        #     weight_decay=self.weight_decay if "l2sp" not in self.loss_mode else 0.0,
        # )

        def lambda_schedule(epoch: int) -> float:
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

        if self.lr_schedule != "reduce_on_plateau":
            lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda_schedule,
            )
            if self.stepwise_schedule:
                lr_scheduler = {
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

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Please implement this method in your subclass for non-square resizes,
        normalizations, etc. To apply nothing, return the identity function
            `lambda x: x`.
        Note that for square resizes we have the `data_resize_transform` argument
        in the `TrainingArgs` class. This is a special case worth supporting
        because it allows easily switching between little image MNIST and large
        image non-MNIST Datasets. Setting it to `Null` / `None` will give you
        full control here.
        """
        return lambda x: x

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        """Add your data augmentations here. Function will be called after get_tensor_transforms in the training loop"""
        return lambda x: x


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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Resize((224), antialias=True)


class VisionTransformerWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_large_patch16_224", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Compose(
            [
                transforms.Resize((224), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class VisionTransformerDinoV2Wrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch)
        # self.model.reset_classifier(self.embedding_size) # TODO
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Compose(
            [
                transforms.Resize((518), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.fc.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )

        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Compose(
            [
                transforms.Resize((224), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.Resize((192), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.Resize((192), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.Resize((192), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
        swin_model = "swinv2_large_window12_192.ms_in22k"
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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.Resize((192), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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


class FaceNetWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = InceptionResnetV1(pretrained="vggface2")

        self.model.last_linear = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1792),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=1792, out_features=self.embedding_size),
        )
        self.model.last_bn = torch.nn.BatchNorm1d(self.embedding_size)
        self.set_losses(self.model, **kwargs)

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.Resize((192), antialias=True),
                transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
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
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        # return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # TODO (rob2u): might be necessary to remove this for wildme model finetuning
        return lambda x: x

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
    "FaceNet": FaceNetWrapper,
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
