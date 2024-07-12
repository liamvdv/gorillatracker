import copy
from typing import Any, Callable

import timm
import torch
import torchvision.transforms.v2 as transforms_v2
from print_on_steroids import logger
from torch import nn
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

from gorillatracker.model.base_module import BaseModule
from gorillatracker.model.model_miewid import GeM, load_miewid_model  # type: ignore


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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.classifier = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.classifier = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)


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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=192
        )
        self.model.reset_classifier(self.embedding_size)
        self.model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.head.in_features),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features=self.model.head.in_features, out_features=self.embedding_size),
            torch.nn.BatchNorm1d(self.embedding_size),
        )
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)
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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.head.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.13)),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(256, scale=(0.75, 1.0)),
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

        model_cpy = copy.deepcopy(self.model)
        model_cpy.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
        model_cpy = copy.deepcopy(self.model)
        model_cpy.fc = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)

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
