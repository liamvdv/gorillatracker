import importlib
from typing import Any, Tuple, Type, Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

import gorillatracker.type_helper as gtypes
from gorillatracker.data_modules import QuadletDataModule, SimpleDataModule, TripletDataModule


def get_dataset_class(pypath: str) -> Type[Dataset[Tuple[torch.Tensor, Union[str, int]]]]:
    parent = torch.utils.data.Dataset
    modpath, clsname = pypath.rsplit(".", 1)
    mod = importlib.import_module(modpath)
    cls = getattr(mod, clsname)
    assert issubclass(cls, parent), f"{cls} is not a subclass of {parent}"
    return cls


def _assert_tensor(x: Any) -> torch.Tensor:
    assert isinstance(
        x, torch.Tensor
    ), f"GorillaTrackerDataset.get_transforms must contain ToTensor. Transformed result is {type(x)}"
    return x


def get_data_module(  # TODO(rob2u): add simple data module
    dataset_class_id: str,
    data_dir: str,
    batch_size: int,
    loss_mode: str,
    model_transforms: gtypes.Transform,
    training_transforms: gtypes.Transform = None,  # type: ignore
) -> Union[TripletDataModule, QuadletDataModule, SimpleDataModule]:
    base = QuadletDataModule if loss_mode.startswith("online") else None
    base = TripletDataModule if loss_mode.startswith("offline") else base  # type: ignore
    base = SimpleDataModule if loss_mode.startswith("softmax") else base  # type: ignore
    dataset_class = get_dataset_class(dataset_class_id)
    dataset_transforms = Compose(
        [
            dataset_class.get_transforms() if hasattr(dataset_class, "get_transforms") else ToTensor(),
            # _assert_tensor,
        ]
    )

    return base(data_dir, batch_size, dataset_class, dataset_transforms=dataset_transforms, model_transforms=model_transforms, training_transforms=training_transforms)  # type: ignore
