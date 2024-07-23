from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Type

import lightning as L
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import (
    ContrastiveClassSampler,
    ContrastiveImage,
    ContrastiveSampler,
    SupervisedCrossEncounterSampler,
    SupervisedHardCrossEncounterSampler,
    ContrastiveKFoldValSampler,
    get_individual,
    group_contrastive_images,
)
from gorillatracker.transform_utils import SquarePad
from gorillatracker.type_helper import Label, Nlet
from gorillatracker.utils.labelencoder import LabelEncoder

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

    # NOTE(memben): The dataloader batches like:
    # batched_ids = ((ap1, ap2, ap3), (p1, p2, p3), ...)
    # batched_values = torch.Tensor((ap1, ap2, ap3), (p1, p2, p3), ...)
    # batched_labels = torch.Tensor((ap1, ap2, ap3), (p1, p2, p3), ...)
    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        if not hasattr(
            self, "train"
        ):  # HACK(rob2u): we enforce setup to be called (somehow it's not always called, problem in val_before_training)
            self.setup("fit")
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers) for val in self.val
        ]

    def test_dataloader(self) -> list[DataLoader[gtypes.Nlet]]:
        return [
            DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers) for test in self.test
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


def group_images_by_label(dirpath: Path) -> defaultdict[Label, list[ContrastiveImage]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.png
            or
            <label>_<...>.jpg
    """
    assert os.path.exists(dirpath), f"Directory {dirpath} does not exist"

    samples = []
    image_paths = list(dirpath.glob("*.jpg"))
    image_paths = image_paths + list(dirpath.glob("*.png"))
    for image_path in image_paths:
        if "_" in image_path.name:
            label = get_individual(image_path)  # type: ignore
        else:
            label = image_path.name.split("-")[0]
        samples.append(ContrastiveImage(str(image_path), image_path, LabelEncoder.encode(label)))
    return group_contrastive_images(samples)

def group_images_by_fold_and_label(dirpaths: list[Path]) -> defaultdict[Label, list[ContrastiveImage]]:
    """
    Assumed directory structure:
        dirpath/
            <label>_<...>.png
            or
            <label>_<...>.jpg
    """
    for dirpath in dirpaths:
        assert os.path.exists(dirpath), f"Directory {dirpath} does not exist"
    samples = []
    image_paths = []
    for dirpath in dirpaths:
        image_paths = image_paths + list(dirpath.glob("*.jpg"))
        image_paths = image_paths + list(dirpath.glob("*.png"))
    for image_path in image_paths:
        fold = image_path.parent.name
        if "_" in image_path.name:
            label = fold + image_path.name.split("_")[0]
        else:
            label = fold + image_path.name.split("-")[0]
        samples.append(ContrastiveImage(str(image_path), image_path, LabelEncoder.encode(label)))
    return group_contrastive_images(samples)


class SupervisedDataset(NletDataset):
    """
    A dataset that assumes the following directory structure:
        data_dir/
            train/
                ...
            val/
                ...
            test/
                ...
    Each file is prefixed with the class label, e.g. "label1_1.jpg"
    """

    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler.class_labels)

    @property
    def class_distribution(self) -> dict[Label, int]:
        return {label: len(samples) for label, samples in self.classes.items()}

    def create_contrastive_sampler(
        self, base_dir: Path, sampler_class: type = ContrastiveClassSampler
    ) -> ContrastiveClassSampler:
        """
        Assumes directory structure:
            data_dir/
                train/
                    ...
                val/
                    ...
                test/
                    ...
        """
        dirpath = base_dir / Path(self.partition) if os.path.exists(base_dir / Path(self.partition)) else base_dir
        assert os.path.exists(dirpath), f"Directory {dirpath} does not exist"
        self.classes = group_images_by_label(dirpath)
        return sampler_class(self.classes)


class SupervisedKFoldDataset(KFoldNletDataset):
    @property
    def num_classes(self) -> int:
        return len(self.contrastive_sampler.class_labels)

    @property
    def class_distribution(self) -> dict[Label, int]:
        return {label: len(samples) for label, samples in self.classes.items()}

    def create_contrastive_sampler(
        self, base_dir: Path, sampler_class: type = ContrastiveClassSampler
    ) -> ContrastiveClassSampler:
        """
        Assumes directory structure:
            data_dir/
                fold-0/
                    ...
                fold-(k-1)/
                    ...
                test/
                    ...
        """
        if self.partition == "train":
            self.classes: defaultdict[Label, list[ContrastiveImage]] = defaultdict(list)
            for i in range(self.k):
                if i == self.val_i:
                    continue
                dirpath = base_dir / Path(f"fold-{i}")
                new_classes = group_images_by_label(dirpath)
                for label, samples in new_classes.items():
                    # NOTE(memben): NO deduplication here (feel free to add it)
                    self.classes[label].extend(samples)
        elif self.partition == "test":
            dirpath = base_dir / Path(self.partition)
            self.classes = group_images_by_label(dirpath)
        elif self.partition == "val":
            dirpath = base_dir / Path(f"fold-{self.val_i}")
            self.classes = group_images_by_label(dirpath)
        else:
            raise ValueError(f"Invalid partition: {self.partition}")
        return sampler_class(self.classes)
    
    
class ValOnlyKfoldDataset(SupervisedDataset):
    def create_contrastive_sampler(
        self, base_dir: Path, sampler_class: Type = ContrastiveKFoldValSampler
    ) -> ContrastiveClassSampler:
        assert self.partition == "val", "ValOnlyKfoldDataset is only for additional validation datasets"
        self.k = len(os.listdir(base_dir)) -1 # subtract 1 for test 
        dirpaths = [base_dir / Path(f"fold-{i}") for i in range(self.k)]
        self.classes = group_images_by_fold_and_label(dirpaths)   
        return sampler_class(self.classes, self.k)    
            

class CrossEncounterSupervisedDataset(SupervisedDataset):
    """Ensure that the positive sample is always from a different video except there is only one video present in the dataset."""

    def __init__(self, data_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=data_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = self.create_contrastive_sampler(
            data_dir, sampler_class=SupervisedCrossEncounterSampler
        )


class CrossEncounterSupervisedKFoldDataset(SupervisedKFoldDataset):
    """Ensure that the positive sample is always from a different video except there is only one video present in the dataset."""

    def __init__(self, data_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=data_dir, *args, **kwargs)
        self.contrastive_sampler = self.create_contrastive_sampler(
            data_dir, sampler_class=SupervisedCrossEncounterSampler
        )


class HardCrossEncounterSupervisedKFoldDataset(SupervisedKFoldDataset):
    """Ensure that the positive sample is always from a different video and discard samples where only one video is present."""

    def __init__(self, data_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(data_dir=data_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = self.create_contrastive_sampler(
            data_dir, sampler_class=SupervisedHardCrossEncounterSampler
        )  # TODO


class HardCrossEncounterSupervisedDataset(SupervisedDataset):
    """Ensure that the positive sample is always from a different video and discard samples where only one video is present."""

    def __init__(self, data_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=data_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = self.create_contrastive_sampler(
            data_dir, sampler_class=SupervisedHardCrossEncounterSampler
        )


def build_onelet(idx: int, contrastive_sampler: ContrastiveSampler) -> tuple[ContrastiveImage]:
    return (contrastive_sampler[idx],)


def build_pair(idx: int, contrastive_sampler: ContrastiveSampler) -> tuple[ContrastiveImage, ContrastiveImage]:
    anchor_positive = contrastive_sampler[idx]
    positive = contrastive_sampler.positive(anchor_positive)
    return (anchor_positive, positive)


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
