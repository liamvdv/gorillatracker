from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Type

import lightning as L
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import ContrastiveImage, ContrastiveSampler
from gorillatracker.transform_utils import SquarePad
from gorillatracker.type_helper import Nlet

FlatNlet = tuple[ContrastiveImage, ...]


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NletDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_class: Type[NletDataset],
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        batch_size: int = 32,
        transforms: gtypes.TensorTransform = lambda x: x,
        training_transforms: gtypes.TensorTransform = lambda x: x,
        eval_datasets: list[Type[NletDataset]] = [],
        eval_data_dirs: list[Path] = [],
        **kwargs: dict[str, Any],  # KFold args, SSLConfig, etc.
    ) -> None:
        """
        The `eval_datasets` are used for evaluation purposes and are additional to the primary `dataset_class`.
        """
        super().__init__()
        assert len(eval_datasets) == len(eval_data_dirs), "eval_datasets and eval_data_dirs must have the same length"
        assert (
            dataset_class not in eval_datasets
        ), "dataset_class should not be in eval_datasets, as it will be added automatically"

        self.data_dir = data_dir
        self.dataset_class = dataset_class
        self.nlet_builder = nlet_builder
        self.batch_size = batch_size
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.eval_datasets = [dataset_class] + eval_datasets
        self.eval_data_dirs = [data_dir] + eval_data_dirs
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        assert stage in {"fit", "validate", "test", "predict"}

        if stage == "fit":
            self.train = self.dataset_class(
                self.data_dir,
                nlet_builder=self.nlet_builder,
                partition="train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
                **self.kwargs,
            )

        if stage == "fit" or stage == "validate":
            self.val = [
                dataset_class(
                    data_dir, nlet_builder=self.nlet_builder, partition="val", transform=self.transforms, **self.kwargs
                )
                for dataset_class, data_dir in zip(self.eval_datasets, self.eval_data_dirs)
            ]

        if stage == "test":
            self.test = [
                dataset_class(
                    data_dir, nlet_builder=self.nlet_builder, partition="test", transform=self.transforms, **self.kwargs
                )
                for dataset_class, data_dir in zip(self.eval_datasets, self.eval_data_dirs)
            ]

        if stage == "predict":
            raise NotImplementedError("Predict not implemented")

    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=100
        )

    def val_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(val, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=100)
            for val in self.val
        ]

    def test_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=100)
            for test in self.test
        ]

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        raise NotImplementedError

    def collate_fn(self, batch: list[gtypes.Nlet]) -> gtypes.NletBatch:
        ids = tuple(nlet[0] for nlet in batch)
        values = tuple(nlet[1] for nlet in batch)
        labels = tuple(nlet[2] for nlet in batch)
        return ids, values, labels

    # TODO(memben): we probably want tuple[int, list[int], list[int]]
    def get_num_classes(self) -> tuple[int, int, int]:
        return (-1, -1, -1)


class NletDataset(Dataset[Nlet], ABC):
    def __init__(
        self,
        base_dir: Path,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        transform: gtypes.TensorTransform,
        **kwargs: dict[str, Any],
    ):
        self.contrastive_sampler = self.create_contrastive_sampler(base_dir)
        self.nlet_builder = nlet_builder
        self.transform: Callable[[Image], torch.Tensor] = transforms.Compose([self.get_transforms(), transform])
        self.partition = partition

    def __len__(self) -> int:
        return len(self.contrastive_sampler)

    def __getitem__(self, idx: int) -> Nlet:
        # NOTE(memben): We want to cache the nlets for the validation and test sets
        if self.partition in {"val", "test"}:
            return self._get_cached_item(idx)
        else:
            return self._get_item(idx)

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @abstractmethod
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveSampler:
        pass

    @lru_cache(maxsize=None)
    def _get_cached_item(self, idx: int) -> Nlet:
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Nlet:
        flat_nlet = self.nlet_builder(idx, self.contrastive_sampler)
        return self.stack_flat_nlet(flat_nlet)

    def stack_flat_nlet(self, flat_nlet: FlatNlet) -> Nlet:
        ids = tuple(str(img.image_path) for img in flat_nlet)
        labels = tuple(img.class_label for img in flat_nlet)
        values = tuple(self.transform(img.image) for img in flat_nlet)
        return ids, values, labels

    @classmethod
    def get_transforms(cls) -> gtypes.Transform:
        return transforms.Compose(
            [
                SquarePad(),
                transforms.ToTensor(),
            ]
        )


class KFoldNletDataset(NletDataset):
    def __init__(
        self,
        data_dir: Path,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        val_i: int,
        k: int,
        transform: gtypes.TensorTransform,
    ):
        super().__init__(data_dir, nlet_builder, partition, transform)
        assert val_i < k, "val_i must be less than k"
        self.val_i = val_i
        self.k = k


def build_triplet(
    idx: int, contrastive_sampler: ContrastiveSampler
) -> tuple[ContrastiveImage, ContrastiveImage, ContrastiveImage]:
    anchor_positive = contrastive_sampler[idx]
    positive = contrastive_sampler.positive(anchor_positive)
    negative = contrastive_sampler.negative(anchor_positive)
    return anchor_positive, positive, negative


def build_quadlet(
    idx: int, contrastive_sampler: ContrastiveSampler
) -> tuple[ContrastiveImage, ContrastiveImage, ContrastiveImage, ContrastiveImage]:
    anchor_positive = contrastive_sampler[idx]
    positive = contrastive_sampler.positive(anchor_positive)
    anchor_negative = contrastive_sampler.negative(anchor_positive)
    negative = contrastive_sampler.positive(anchor_negative)
    return anchor_positive, positive, anchor_negative, negative
