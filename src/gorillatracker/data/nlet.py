from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Type

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


class FlatNletBuilder(Protocol):
    def __call__(self, idx: int, contrastive_sampler: ContrastiveSampler) -> FlatNlet: ...


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NletDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_class: Type[NletDataset],
        nlet_builder: FlatNletBuilder,
        batch_size: int,
        workers: int,
        model_transforms: gtypes.TensorTransform,
        training_transforms: gtypes.TensorTransform,
        eval_datasets: list[Type[NletDataset]] = [],
        eval_data_dirs: list[Path] = [],
        **kwargs: Any,  # SSLConfig, etc.
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
        self.workers = workers
        self.model_transforms = model_transforms
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
                transform=transforms.Compose([self.model_transforms, self.training_transforms]),
                **self.kwargs,
            )

        if stage == "fit" or stage == "validate":
            self.val = [
                dataset_class(
                    data_dir,
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

    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.workers
        )

    def val_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(
                val, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=self.workers
            )
            for val in self.val
        ]

    def test_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(
                test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=self.workers
            )
            for test in self.test
        ]

    def predict_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:  # TODO(memben)
        raise NotImplementedError

    def collate_fn(self, batch: list[gtypes.Nlet]) -> gtypes.NletBatch:
        ids = tuple(nlet[0] for nlet in batch)
        values = tuple(nlet[1] for nlet in batch)
        labels = tuple(nlet[2] for nlet in batch)
        batched_ids = tuple(zip(*ids))
        batched_values = tuple(zip(*values))
        batched_labels = tuple(zip(*labels))
        return batched_ids, batched_values, batched_labels

    def get_dataset_class_names(self) -> list[str]:
        return [dataset_class.__name__ for dataset_class in self.eval_datasets]

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


class NletDataset(Dataset[Nlet], ABC):
    def __init__(
        self,
        base_dir: Path,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        transform: gtypes.TensorTransform,
        **kwargs: Any,
    ):
        self.partition = partition
        self.contrastive_sampler = self.create_contrastive_sampler(base_dir)
        self.nlet_builder = nlet_builder
        self.transform: Callable[[Image], torch.Tensor] = transforms.Compose([self.get_transforms(), transform])

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

    @property
    @abstractmethod
    def class_distribution(self) -> dict[gtypes.Label, int]:
        pass

    @abstractmethod
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveSampler:
        pass

    @lru_cache(maxsize=None)
    def _get_cached_item(self, idx: int) -> Nlet:
        return self._get_item(idx)

    def _get_item(self, idx: int) -> Nlet:
        flat_nlet = self.nlet_builder(idx, self.contrastive_sampler)
        return self._stack_flat_nlet(flat_nlet)

    def _stack_flat_nlet(self, flat_nlet: FlatNlet) -> Nlet:
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
        **kwargs: Any,
    ):
        assert val_i < k, "val_i must be less than k"
        self.k = k
        self.val_i = val_i
        super().__init__(data_dir, nlet_builder, partition, transform)


def build_onelet(idx: int, contrastive_sampler: ContrastiveSampler) -> tuple[ContrastiveImage]:
    return (contrastive_sampler[idx],)


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
