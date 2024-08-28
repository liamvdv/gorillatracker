from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import (
    ContrastiveClassSampler,
    ContrastiveImage,
    ContrastiveKFoldValSampler,
    ContrastiveSampler,
    FlatNlet,
    SupervisedCrossEncounterSampler,
    SupervisedHardCrossEncounterSampler,
    get_individual,
    group_contrastive_images,
)
from gorillatracker.transform_utils import SquarePad
from gorillatracker.type_helper import Label, Nlet
from gorillatracker.utils.labelencoder import LabelEncoder


class FlatNletBuilder(Protocol):
    def __call__(self, idx: int, contrastive_sampler: ContrastiveSampler) -> FlatNlet: ...


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.transform: Callable[[Image], torch.Tensor] = transforms.Compose([self.get_transforms(partition, kwargs["aug_num_ops"], kwargs["aug_magnitude"]), transform])

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
    def get_transforms(cls, partition: str = "train", num_ops: int = 2, magnitude: int = 9) -> gtypes.Transform:
        if partition != "train":
            return transforms.Compose(
                [
                    SquarePad(),
                    transforms.ToTensor(),
                ]
            )
        elif num_ops == 0:
            return transforms.Compose(
                [
                    SquarePad(),
                    transforms.ToTensor(),
                ]
            )
        else:
            return transforms.Compose(
                [
                    SquarePad(),
                    transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
                    transforms.ToTensor(),
                ]
            )
        


class KFoldNletDataset(NletDataset):
    def __init__(
        self,
        base_dir: Path,
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
        super().__init__(base_dir, nlet_builder, partition, transform, **kwargs)


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
    image_paths: list[Path] = []
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
        base_dir/
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
            base_dir/
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
            base_dir/
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


class ValOnlyKFoldDataset(SupervisedDataset):
    def create_contrastive_sampler(
        self, base_dir: Path, sampler_class: type = ContrastiveKFoldValSampler
    ) -> ContrastiveClassSampler:
        assert self.partition == "val", "ValOnlyKfoldDataset is only for additional validation datasets"
        self.k = len(os.listdir(base_dir)) - 1  # subtract 1 (test set as its not a fold)
        dirpaths = [base_dir / Path(f"fold-{i}") for i in range(self.k)]
        self.classes = group_images_by_fold_and_label(dirpaths)
        return sampler_class(self.classes, self.k)


class CrossEncounterSupervisedDataset(SupervisedDataset):
    """Ensure that the positive sample is always from a different video except there is only one video present in the dataset."""

    def __init__(self, base_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=base_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = (
            self.create_contrastive_sampler(base_dir, sampler_class=SupervisedCrossEncounterSampler)
            if self.partition == "train"
            else self.create_contrastive_sampler(base_dir)
        )


class CrossEncounterSupervisedKFoldDataset(SupervisedKFoldDataset):
    """Ensure that the positive sample is always from a different video except there is only one video present in the dataset."""

    def __init__(self, base_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=base_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = (
            self.create_contrastive_sampler(base_dir, sampler_class=SupervisedCrossEncounterSampler)
            if self.partition == "train"
            else self.create_contrastive_sampler(base_dir)
        )


class HardCrossEncounterSupervisedKFoldDataset(SupervisedKFoldDataset):
    """Ensure that the positive sample is always from a different video and discard samples where only one video is present."""

    def __init__(self, base_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=base_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = (
            self.create_contrastive_sampler(base_dir, sampler_class=SupervisedHardCrossEncounterSampler)
            if self.partition == "train"
            else self.create_contrastive_sampler(base_dir)
        )


class HardCrossEncounterSupervisedDataset(SupervisedDataset):
    """Ensure that the positive sample is always from a different video and discard samples where only one video is present."""

    def __init__(self, base_dir: Path, *args: Any, **kwargs: Any):
        super().__init__(base_dir=base_dir, *args, **kwargs)  # type: ignore
        self.contrastive_sampler = (
            self.create_contrastive_sampler(base_dir, sampler_class=SupervisedHardCrossEncounterSampler)
            if self.partition == "train"
            else self.create_contrastive_sampler(base_dir)
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
