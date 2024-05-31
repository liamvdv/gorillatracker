import importlib
from typing import Any, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

import gorillatracker.type_helper as gtypes
from gorillatracker.data_modules import (
    NletDataModule,
    QuadletDataModule,
    QuadletKFoldDataModule,
    SimpleDataModule,
    SimpleKFoldDataModule,
    TripletDataModule,
    TripletKFoldDataModule,
)


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


def get_data_module(
    dataset_class_id: str,
    data_dir: str,
    batch_size: int,
    loss_mode: str,
    model_transforms: gtypes.Transform,
    training_transforms: gtypes.Transform = None,  # type: ignore
    additional_dataset_class_ids: Optional[List[str]] = None,
    additional_data_dirs: Optional[List[str]] = None,
) -> NletDataModule:
    base: Type[NletDataModule]
    base = QuadletDataModule if loss_mode.startswith("online") else None  # type: ignore
    base = TripletDataModule if loss_mode.startswith("offline") else base  # type: ignore
    base = SimpleDataModule if loss_mode.startswith("softmax") else base  # type: ignore

    if "kfold" in data_dir:
        base = QuadletKFoldDataModule if loss_mode.startswith("online") else None  # type: ignore
        base = TripletKFoldDataModule if loss_mode.startswith("offline") else base  # type: ignore
        base = SimpleKFoldDataModule if loss_mode.startswith("softmax") else base  # type: ignore

    dataset_class = get_dataset_class(dataset_class_id)
    transforms = Compose(
        [
            dataset_class.get_transforms() if hasattr(dataset_class, "get_transforms") else ToTensor(),
            _assert_tensor,
            model_transforms,
        ]
    )
    if additional_dataset_class_ids is None:
        return base(data_dir, batch_size, dataset_class, transforms=transforms, training_transforms=training_transforms)
    else:
        assert additional_data_dirs is not None, "additional_data_dirs must be set"
        assert "kfold" not in data_dir, "kfold not supported for additional datasets"
        dataset_classes = [get_dataset_class(cls_id) for cls_id in additional_dataset_class_ids]
        transforms_list = []
        for cls in dataset_classes:
            transforms_list.append(
                Compose(
                    [
                        cls.get_transforms() if hasattr(cls, "get_transforms") else ToTensor(),
                        _assert_tensor,
                        model_transforms,
                    ]
                )
            )

        return base(
            data_dir,
            batch_size,
            dataset_class,
            transforms=transforms,
            training_transforms=training_transforms,
            additional_dataset_classes=dataset_classes,
            additional_data_dirs=additional_data_dirs,
            additional_transforms=transforms_list,
        )
