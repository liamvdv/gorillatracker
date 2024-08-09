from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Literal, Optional, Type

import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from timm.layers.classifier import ClassifierHead, NormMlpClassifierHead
from torchvision import transforms
from transformers import AutoModel, ResNetModel

from gorillatracker.model.base_module import BaseModule
from gorillatracker.model.pooling_layers import GAP, FormatWrapper, GeM, GeM_adapted
from gorillatracker.model.wrapper_mae import MaskedVisionTransformer
from gorillatracker.transform_utils import PlanckianJitter

logger = getLogger(__name__)


def get_global_pooling_layer(id: str, num_features: int, format: Literal["NCHW", "NHWC"] = "NCHW") -> torch.nn.Module:
    if id == "gem":
        return FormatWrapper(GeM(), format)
    elif id == "gem_c":
        return FormatWrapper(GeM_adapted(p_shape=(num_features)), format)  # TODO(rob2u): test
    elif id == "gap":
        return FormatWrapper(GAP(), format)
    else:
        return nn.Identity()


def get_embedding_layer(id: str, feature_dim: int, embedding_dim: int, dropout_p: float = 0.0) -> torch.nn.Module:
    if id == "linear":
        return nn.Linear(feature_dim, embedding_dim)
    elif id == "mlp":
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, embedding_dim),
        )
    elif "linear_norm_dropout" in id:
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    elif "mlp_norm_dropout" in id:
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    else:
        return nn.Identity()


# NOTE(rob2u): We used the following models from timm:
# efficientnetv2_rw_m — EfficientNetRW_M
# convnextv2_base — ConvNeXtV2BaseWrapper
# convnextv2_huge — ConvNeXtV2HugeWrapper
# vit_large_patch16_224 — VisionTransformerWrapper
# vit_large_patch14_dinov2.lvd142m — VisionTransformerDinoV2Wrapper
# vit_base_patch16_clip_224.metaclip_2pt5b — VisionTransformerClipWrapper
# convnext_base.clip_laion2b — ConvNextClipWrapper
# swinv2_base_window12_192.ms_in22k — SwinV2BaseWrapper
# swinv2_large_window12to16_192to256.ms_in22k_ft_in1k — SwinV2LargeWrapper
# inception_v3 — InceptionV3Wrapper
class TimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        pool_mode: Literal["gem", "gap", "gem_c", "none"] = "none",
        img_size: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert pool_mode == "none" or "vit" not in backbone_name, "pool_mode is not supported for VisionTransformer."
        if img_size is not None:
            logger.info("Setting img_size to", img_size)
            self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0, img_size=img_size)
        else:
            self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0)
        self.num_features = self.model.num_features

        self.reset_if_necessary(pool_mode)
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=self.num_features, embedding_dim=embedding_size, dropout_p=dropout_p
        )
        self.pool_mode = pool_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.dim() == 3:
            logger.info("Assuming VisionTransformer is used and taking the first token.")
            x = x[:, 0, :]

        x = self.embedding_layer(x)
        return x

    def reset_if_necessary(self, pool_mode: Optional[Literal["gem", "gap", "gem_c", "none"]] = None) -> None:
        if pool_mode == "none":
            pool_mode = None
        if (
            hasattr(self.model, "head")
            # NOTE(rob2u): see https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/classifier.py#L73
            and hasattr(self.model.head, "global_pool")
            and pool_mode is not None
        ):
            if isinstance(self.model.head, ClassifierHead):
                self.model.head.global_pool = get_global_pooling_layer(pool_mode, self.model.head.input_fmt)
                self.model.head.fc = nn.Identity()
                self.model.head.drop = nn.Identity()
            elif isinstance(self.model.head, NormMlpClassifierHead):
                logger.warn(
                    "Model uses NormMlpClassifierHead, for which we do not want to change the global_pooling layer."
                )
        elif pool_mode is not None and hasattr(self.model, "global_pool"):
            self.model.reset_classifier(0, "")
            self.model.global_pool = get_global_pooling_layer(pool_mode, self.num_features)
        else:
            logger.info("No pooling layer reset necessary.")


class TimmEvalWrapper(nn.Module):
    def __init__(  # type: ignore
        self,
        backbone_name,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=True)
        if timm.data.resolve_model_data_config(self.model)["input_size"][-1] > 768:
            logger.warn("We wont use image size greater than 768!!!")
            self.model = timm.create_model(backbone_name, pretrained=True, img_size=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        return x


class ResNet50DinoV2Wrapper(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=2048, embedding_dim=embedding_size, dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        feature_vector = gap(outputs.last_hidden_state)
        feature_vector = torch.flatten(feature_vector, start_dim=2).squeeze(-1)
        feature_vector = self.embedding_layer(feature_vector)
        return feature_vector


class Miewid_msv2(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True)  # size: 440
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=1280, embedding_dim=embedding_size, dropout_p=dropout_p
        )  # TODO(rob2u): test

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.embedding_layer(x)
        return x


class MAEFineTuningWrapper(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        mae = MaskedVisionTransformer.load_from_checkpoint(checkpoint_path, data_module=None, wandb_run=None)
        self.model = mae.backbone.vit  # NOTE(rob2u): a timm model
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=self.model.num_features, embedding_dim=embedding_size, dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.dim() == 3:
            logger.info("Assuming VisionTransformer is used and taking the first token.")
            x = x[:, 0, :]

        x = self.embedding_layer(x)
        return x


model_wrapper_registry = {
    "timm": TimmWrapper,
    "timm_eval": TimmEvalWrapper,
    "resnet50_dinov2": ResNet50DinoV2Wrapper,
    "miewid_msv2": Miewid_msv2,
    "MAE": MAEFineTuningWrapper,
}


class BaseModuleSupervised(BaseModule):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_p: float = 0.0,
        pool_mode: Optional[Literal["gem", "gap", "gem_c"]] = None,
        fix_img_size: Optional[int] = None,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        freeze_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        assert (
            len(model_name_or_path.split("/")) >= 2
        ), "model_name_or_path should be in the format '[<wrapper_id>/]<model_id>'."
        logger.info("Using model", model_name_or_path)
        wrapper_cls: Type[nn.Module] = model_wrapper_registry.get(model_name_or_path.split("/")[0], TimmWrapper)
        if model_name_or_path.startswith("timm") or model_name_or_path.startswith(
            "timm_eval"
        ):  # Example: hf-hub:BVRA/MegaDescriptor-T-224  # Example: timm/efficientnetv2_rw_m
            backbone_name = model_name_or_path.split("/")[-1]
        else:
            backbone_name = model_name_or_path

        self.model_wrapper = wrapper_cls(
            backbone_name=backbone_name,
            pool_mode=pool_mode,
            img_size=fix_img_size,
            embedding_size=self.embedding_size,
            embedding_id=embedding_id,
            dropout_p=dropout_p,
            checkpoint_path=(
                "/".join(model_name_or_path.split("/")[1:]) if model_name_or_path.startswith("MAE") else None
            ),
        )
        self.set_losses(model=self.model_wrapper.model, **kwargs)

        if freeze_backbone:
            for param in self.model_wrapper.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # TODO check for l2sp
        x = self.model_wrapper(x)
        return x

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                PlanckianJitter(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                # transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )
