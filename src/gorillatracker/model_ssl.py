import copy
from typing import Callable

import timm
import torch
from lightly.models.modules.heads import SimCLRProjectionHead, MoCoProjectionHead
from lightly.models.utils import (
    deactivate_requires_grad,
    update_momentum,
)
import torchvision.transforms.v2 as transforms_v2
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.utils import flatten_batch
from gorillatracker.model import BaseModule

class SimClRWrapper(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=224)
        self.model.reset_classifier(self.embedding_size)
        in_features = self.model.head.in_features
        self.model.head = SimCLRProjectionHead(in_features, in_features, self.embedding_size)
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
