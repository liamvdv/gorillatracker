from typing import Type

from gorillatracker.model.base_module import BaseModule
from gorillatracker.model.wrappers_ssl import MoCoWrapper, SimCLRWrapper
from gorillatracker.model.wrappers_supervised import BaseModuleSupervised

# NOTE(liamvdv): Register custom model backbones here.
custom_model_cls = {
    "SimCLR": SimCLRWrapper,
    "MoCo": MoCoWrapper,
}


def get_model_cls(model_name: str) -> Type[BaseModule]:
    model_cls: Type[BaseModule]
    model_cls = custom_model_cls.get(model_name, BaseModuleSupervised)
    return model_cls
