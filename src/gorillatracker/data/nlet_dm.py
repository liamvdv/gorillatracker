from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Protocol, Type, Union

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.combined import CombinedDataset
from gorillatracker.data.contrastive_sampler import ContrastiveSampler, FlatNlet
from gorillatracker.data.nlet import NletDataset


class FlatNletBuilder(Protocol):
    def __call__(self, idx: int, contrastive_sampler: ContrastiveSampler) -> FlatNlet: ...


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NletDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_class: Union[Type[NletDataset], Type[CombinedDataset]],
        nlet_builder: FlatNletBuilder,
        batch_size: int,
        workers: int,
        model_transforms: gtypes.TensorTransform,
        training_transforms: gtypes.TensorTransform,
        eval_datasets: list[Union[Type[NletDataset], Type[CombinedDataset]]] = [],
        eval_data_dirs: list[Path] = [],
        dataset_ids: list[str] = [],
        dataset_names: list[str] = [],
        **kwargs: Any,  # SSLConfig, etc.
    ) -> None:
        """
        The `eval_datasets` are used for evaluation purposes and are additional to the primary `dataset_class`.
        """
        super().__init__()
        assert len(eval_datasets) == len(eval_data_dirs), "eval_datasets and eval_data_dirs must have the same length"
        assert (
            len(eval_datasets) == len(dataset_names) - 1
        ), "eval_datasets and eval_dataset_names must have the same length"

        self.data_dir = data_dir
        self.dataset_class = dataset_class
        self.nlet_builder = nlet_builder
        self.batch_size = batch_size
        self.workers = workers
        self.model_transforms = model_transforms
        self.training_transforms = training_transforms
        self.eval_datasets = [dataset_class] + eval_datasets
        self.eval_data_dirs = [data_dir] + eval_data_dirs
        self.dataset_ids = dataset_ids
        self.dataset_names = dataset_names
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        assert stage in {"fit", "validate", "test", "predict"}
        if stage == "fit":
            self.train = self.dataset_class(
                self.data_dir,
                nlet_builder=self.nlet_builder,
                partition="train",
                transform=transforms.Compose([self.training_transforms, self.model_transforms]),
                **self.kwargs,
            )

        if stage == "fit" or stage == "validate":
            self.val = [
                dataset_class(
                    base_dir=data_dir,
                    nlet_builder=self.nlet_builder,
                    partition="val",
                    transform=self.model_transforms,
                    **self.kwargs,
                )
                for dataset_class, data_dir in zip(self.eval_datasets, self.eval_data_dirs)
            ]

        if stage == "test":
            self.test = [
                dataset_class(
                    data_dir,
                    nlet_builder=self.nlet_builder,
                    partition="test",
                    transform=self.model_transforms,
                    **self.kwargs,
                )
                for dataset_class, data_dir in zip(self.eval_datasets, self.eval_data_dirs)
            ]

        if stage == "predict":
            raise NotImplementedError("Predict not implemented")

    # NOTE(memben): The dataloader batches like:
    # batched_ids = ((ap1, ap2, ap3), (p1, p2, p3), ...)
    # batched_values = torch.Tensor((ap1, ap2, ap3), (p1, p2, p3), ...)
    # batched_labels = torch.Tensor((ap1, ap2, ap3), (p1, p2, p3), ...)
    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        if not hasattr(
            self, "train"
        ):  # HACK(rob2u): we enforce setup to be called (somehow it's not always called, problem in val_before_training)
            self.setup("fit")
        # NOTE(rob2u): the type ignores are necessary because these types would be incorrect for the combined Dataset, yet we don't want to change it (cascade of changes)
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, drop_last=True)  # type: ignore

    def val_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        # NOTE(rob2u): the type ignores are necessary because these types would be incorrect for the combined Dataset, yet we don't want to change it (cascade of changes)
        return [
            DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers) for val in self.val  # type: ignore
        ]

    def test_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        # NOTE(rob2u): the type ignores are necessary because these types would be incorrect for the combined Dataset, yet we don't want to change it (cascade of changes)
        return [
            DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers) for test in self.test  # type: ignore
        ]

    def predict_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:  # TODO(memben)
        raise NotImplementedError

    def get_dataset_class_names(self) -> list[str]:
        return self.dataset_names

    def get_dataset_ids(self) -> list[str]:
        return self.dataset_ids

    # TODO(memben): we probably want tuple[int, list[int], list[int]]
    def get_num_classes(self, partition: Literal["train", "val", "test"]) -> int:
        if partition == "train":
            return self.train.num_classes
        elif partition == "val":
            return self.val[0].num_classes
        elif partition == "test":
            return self.test[0].num_classes
        else:
            raise ValueError(f"unknown partition '{partition}'")

    # TODO(memben): we probably want to return a list of dicts
    def get_class_distribution(self, partition: Literal["train", "val", "test"]) -> dict[gtypes.Label, int]:
        if partition == "train":
            return self.train.class_distribution
        elif partition == "val":
            return self.val[0].class_distribution
        elif partition == "test":
            return self.test[0].class_distribution
