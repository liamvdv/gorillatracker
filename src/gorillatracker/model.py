import importlib
from typing import Any, Callable, Literal, Type

import lightning as L
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.v2 as transforms_v2
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
from gorillatracker.triplet_loss import get_triplet_loss


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
        return initial_lr
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
        return initial_lr
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
        beta1: float,
        beta2: float,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        margin: float = 0.5,
        embedding_size: int = 256,
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

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.margin = margin

        # NOTE: Needs to be set by subclasses, cannot use 'None': triggers mypy.
        # self.model = None
        self.from_scratch = from_scratch
        self.embedding_size = embedding_size

        ##### Create Table embeddings_table
        self.embeddings_table_columns = ["label", "embedding"]
        self.embeddings_table = pd.DataFrame(columns=self.embeddings_table_columns)

        # TODO(rob2u): rename loss mode
        self.triplet_loss = get_triplet_loss(loss_mode, margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        vec = torch.cat(images, dim=0)
        embeddings = self.forward(vec)
        flat_labels = (
            torch.cat(labels, dim=0) if torch.is_tensor(labels[0]) else [label for group in labels for label in group]  # type: ignore
        )
        loss, pos_dist, neg_dist = self.triplet_loss(embeddings, flat_labels)  # type: ignore
        self.log("train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/positive_distance", pos_dist, on_step=True)
        self.log("train/negative_distance", neg_dist, on_step=True)
        return loss

    def add_validation_embeddings(self, anchor_embeddings: torch.Tensor, anchor_labels: gtypes.MergedLabels) -> None:
        # save anchor embeddings of validation step for later analysis in W&B
        embeddings = torch.reshape(anchor_embeddings, (-1, self.embedding_size))
        embeddings = embeddings.cpu()

        assert len(self.embeddings_table_columns) == 2
        data = {
            self.embeddings_table_columns[0]: anchor_labels.tolist()  # type: ignore
            if torch.is_tensor(anchor_labels)  # type: ignore
            else anchor_labels,
            self.embeddings_table_columns[1]: [embedding.numpy() for embedding in embeddings],
        }

        df = pd.DataFrame(data)
        self.embeddings_table = pd.concat([df, self.embeddings_table], ignore_index=True)
        # NOTE(rob2u): will get flushed by W&B Callback on val epoch end.

    def validation_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        images, labels = batch  # embeddings either (ap, a, an, n) oder (a, p, n)
        n_achors = len(images[0])
        vec = torch.cat(images, dim=0)
        flat_labels = (
            torch.cat(labels, dim=0) if torch.is_tensor(labels[0]) else [label for group in labels for label in group]  # type: ignore
        )
        embeddings = self.forward(vec)

        self.add_validation_embeddings(embeddings[:n_achors], flat_labels[:n_achors], images[0])  # type: ignore
        loss, pos_dist, neg_dist = self.triplet_loss(embeddings, flat_labels)  # type: ignore
        self.log("val/loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        self.log("val/positive_distance", pos_dist, on_step=True)
        self.log("val/negative_distance", neg_dist, on_step=True)
        return loss

    def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRSchedulerConfig:
        if self.global_rank == 0:
            logger.info(
                f"Using {self.lr_schedule} learning rate schedule with {self.warmup_mode} warmup for {self.max_epochs} epochs."
            )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )

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

            return {"optimizer": optimizer, "lr_scheduler": lambda_scheduler}

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
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.embedding_size),
        )

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ConvNeXtV2BaseWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("convnextv2_base", pretrained=not self.from_scratch)
        self.model.reset_classifier(self.embedding_size)

    def get_grad_cam_layer(self) -> torch.nn.Module:
        return self.model.stages[-1].blocks[-1].conv_dw

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )


class ConvNeXtV2HugeWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("convnextv2_huge", pretrained=not self.from_scratch)
        self.model.reset_classifier(self.embedding_size)

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
        self.model.reset_classifier(self.embedding_size)

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
        self.model.reset_classifier(self.embedding_size)

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.reset_classifier(self.embedding_size)

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
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
        )

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.reset_classifier(self.embedding_size)

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
        )

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
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
        self.model.head.fc = torch.nn.Linear(
            in_features=self.model.head.fc.in_features, out_features=self.embedding_size
        )

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size)

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size)

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.embedding_size)

    @classmethod
    def get_tensor_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
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
        self.last_linear = torch.nn.Linear(in_features=2048, out_features=self.embedding_size)

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
                transforms.RandomErasing(p=0.5, value=(0.707, 0.973, 0.713), scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
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
}


def get_model_cls(model_name: str) -> Type[BaseModule]:
    model_cls = custom_model_cls.get(model_name, None)
    if not model_cls:
        module, cls = model_name.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module), cls)
    return model_cls
