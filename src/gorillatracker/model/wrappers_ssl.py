import copy
from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.utils import flatten_batch
from gorillatracker.model.base_module import BaseModule
from gorillatracker.transform_utils import PlanckianJitter


class SimCLRWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=224
        )
        self.model.reset_classifier(self.embedding_size)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
        )
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                PlanckianJitter(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


# TODO: MoCoWrapper is not fully tested yet.
class MoCoWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=224
        )
        self.model.reset_classifier(self.embedding_size)
        in_features = self.model.head.in_features
        # self.model.head = nn.Sequential(
        #     nn.BatchNorm1d(in_features),
        #     nn.Dropout(p=self.dropout_p),
        #     nn.Linear(in_features, in_features),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(in_features),
        #     nn.Dropout(p=self.dropout_p),
        #     nn.Linear(in_features, self.embedding_size),
        #     nn.BatchNorm1d(self.embedding_size),
        # )
        self.model.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
        )
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

        self.model_momentum = copy.deepcopy(self.model)
        deactivate_requires_grad(self.model_momentum)

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        _, images, _ = batch
        _, flat_images, flat_labels = flatten_batch(batch)

        update_momentum(self.model, self.model_momentum, 0.99)

        flat_labels_onehot = None
        if self.use_inbatch_mixup:
            flat_images, flat_labels_onehot = self.perform_mixup(flat_images, flat_labels)
        anchor_embeddings = self.model(images[0])
        positive_embeddings = self.model_momentum(images[1]).detach()
        embeddings = torch.cat([anchor_embeddings, positive_embeddings])

        assert not torch.isnan(embeddings).any(), f"Embeddings are NaN: {embeddings}"
        loss, pos_dist, neg_dist = self.loss_module_train(embeddings=embeddings, labels=flat_labels, images=images, labels_onehot=flat_labels_onehot)  # type: ignore

        log_str_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""
        self.log(f"{log_str_prefix}train/negative_distance", neg_dist, on_step=True)
        self.log(f"{log_str_prefix}train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log(f"{log_str_prefix}train/positive_distance", pos_dist, on_step=True)
        self.log(f"{log_str_prefix}train/negative_distance", neg_dist, on_step=True)
        return loss

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                PlanckianJitter(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )


class BYOLWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=224
        )
        self.model.reset_classifier(self.embedding_size)
        in_features = self.model.head.in_features
        self.model.head = BYOLProjectionHead(in_features, 1024, self.embedding_size)
        self.prediction_head = BYOLProjectionHead(self.embedding_size, 1024, self.embedding_size)
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)
        self.criterion = NegativeCosineSimilarity()

        self.model_momentum = copy.deepcopy(self.model)
        deactivate_requires_grad(self.model_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        z = self.prediction_head(y)
        return z

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model_momentum(x)
        y = y.detach()
        return y

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        _, images, _ = batch
        _, flat_images, flat_labels = flatten_batch(batch)

        update_momentum(self.model, self.model_momentum, 0.99)

        # Calculate both embeddings and momentum embeddings for anchor and positive images
        anchor_embeddings = self.model(images[0])
        positive_embeddings = self.model(images[1])

        anchor_embeddings_momentum = self.forward_momentum(images[0])
        positive_embeddings_momentum = self.forward_momentum(images[1])

        loss = 0.5 * (
            self.criterion(anchor_embeddings, positive_embeddings_momentum)
            + self.criterion(anchor_embeddings_momentum, positive_embeddings)
        )

        log_str_prefix = f"fold-{self.kfold_k}/" if self.kfold_k is not None else ""
        self.log(f"{log_str_prefix}train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                PlanckianJitter(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )
