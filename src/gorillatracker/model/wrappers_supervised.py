from __future__ import annotations

import copy
from typing import Callable, Literal, Optional

import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from print_on_steroids import logger
from timm.layers.classifier import ClassifierHead, NormMlpClassifierHead
from torchvision import transforms
from transformers import ResNetModel

from gorillatracker.model.base_module import BaseModule
from gorillatracker.model.model_miewid import load_miewid_model  # type: ignore
from gorillatracker.model.pooling_layers import FormatWrapper, GeM, GeM_adapted, GAP



def get_global_pooling_layer(id: str, num_features: int, format: str = "NCHW") -> torch.nn.Module:
    if id == "gem":
        return FormatWrapper(GeM(), format)
    elif id == "gem_c":
        return FormatWrapper(GeM_adapted(p_shape=(1,num_features)), format) # TODO
    elif id == "gap":
        return FormatWrapper(GAP(), format)
    else:
        return nn.Identity()


def get_embedding_layer(id: str, feature_dim: int, embedding_dim: int, dropout_p=0.0) -> torch.nn.Module:
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
            nn.LayerNorm(embedding_dim),
        )
    elif "mlp_norm_dropout" in id:
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
    else:
        return nn.Identity()

 # TODO(rob2u): add freeze option

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
class GenericTimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name,
        embedding_size: int,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        pool_mode: Optional[Literal["gem", "gap", "gem_c"]] = None,
        img_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert pool_mode is None or "vit" not in backbone_name, "pool_mode is not supported for VisionTransformer."
        if img_size is not None:
            print("Setting img_size to", img_size)
            self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0, img_size=img_size)
        self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0)
        self.num_features = self.model.num_features

        self.reset_if_necessary(pool_mode)
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=self.num_features, embedding_dim=embedding_size, dropout_p=dropout_p
        )
        self.pool_mode = pool_mode

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.dim() == 3:
            print("Assuming VisionTransformer is used and taking the first token.")
            x = x[:, 0, :]

        x = self.embedding_layer(x)
        return x

    def reset_if_necessary(self, pool_mode: Optional[Literal["gem", "gap", "gem_c"]] = None):
        if (
            hasattr(self.model, "head")
            # NOTE(rob2u): see https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/classifier.py#L73
            and hasattr(self.model.head, "global_pool")
            and pool_mode is not None
        ):
            if isinstance(self.model.head, ClassifierHead):
                self.model.head.global_pool = get_global_pooling_layer("", self.model.head.input_fmt)
                self.model.head.fc = nn.Identity()
                self.model.head.drop = nn.Identity()
            elif isinstance(self.model.head, NormMlpClassifierHead):
                print("Model uses NormMlpClassifierHead, for which we do not want to change the global_pooling layer.")
        elif pool_mode is not None and hasattr(self.model, "global_pool"):
            self.model.reset_classifier(0, "")
            self.model.global_pool = get_global_pooling_layer("")
        else:
            print("No pooling layer reset necessary.")


class ModuleWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        backbone_name: str,
        pool_mode: Optional[Literal["gem", "gap", "gem_c"]] = None,
        fix_img_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.timm_wrapper = GenericTimmWrapper(backbone_name=backbone_name, pool_mode=pool_mode, img_size=fix_img_size)
        
        self.set_losses(self.timm_wrapper.model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # TODO check for l2sp
        x = self.backbone(x)
        return x

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


class EvaluationWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        model_name_or_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        model_name_or_path = model_name_or_path.replace("timm/", "")
        self.model = timm.create_model(model_name_or_path, pretrained=not self.from_scratch)
        # if timm.data.resolve_model_data_config(self.model)["input_size"][-1] > 768:
        # self.model = timm.create_model(model_name_or_path, pretrained=not self.from_scratch, img_size=512)

        self.set_losses(model=self.model, **kwargs)  # NOTE: necessary for eval (sadly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        return x


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
