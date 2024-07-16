from typing import Optional, Type

from gorillatracker.model.base_module import BaseModule
from gorillatracker.model.wrappers_ssl import MoCoWrapper, SimCLRWrapper
from gorillatracker.model.wrappers_supervised import (
    ConvNextClipWrapper,
    ConvNeXtV2BaseWrapper,
    ConvNeXtV2HugeWrapper,
    ConvNextWrapper,
    EfficientNetRW_M,
    EfficientNetV2Wrapper,
    EvaluationWrapper,
    InceptionV3Wrapper,
    MiewIdNetWrapper,
    ResNet18Wrapper,
    ResNet50DinoV2Wrapper,
    ResNet50Wrapper,
    ResNet152Wrapper,
    SwinV2BaseWrapper,
    SwinV2LargeWrapper,
    VisionTransformerClipWrapper,
    VisionTransformerDinoV2Wrapper,
    VisionTransformerWrapper,
    VisionTransformerFrozenWrapper,
)

# NOTE(liamvdv): Register custom model backbones here.
custom_model_cls = {
    "EfficientNetV2_Large": EfficientNetV2Wrapper,
    "SwinV2Base": SwinV2BaseWrapper,
    "SwinV2LargeWrapper": SwinV2LargeWrapper,
    "ViT_Large": VisionTransformerWrapper,
    "ViT_Large_Frozen": VisionTransformerFrozenWrapper,
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
    "SimCLR": SimCLRWrapper,
    "MoCo": MoCoWrapper,
}


def get_model_cls(model_name: str) -> Type[BaseModule]:
    model_cls: Optional[Type[BaseModule]] = None
    model_cls = custom_model_cls.get(model_name, None)
    if model_cls is None and model_name.startswith("timm/"):
        model_cls = EvaluationWrapper

    assert model_cls is not None, f"Model {model_name} not found in custom_model_cls"
    return model_cls
