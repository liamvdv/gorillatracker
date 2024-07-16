from pathlib import Path
from typing import Any, Callable, Literal, Type

import torch
from PIL import Image
from torchvision import transforms

from gorillatracker import type_helper as gtypes
from gorillatracker.data.contrastive_sampler import ContrastiveImage, ContrastiveSampler, Label
from gorillatracker.data.nlet import FlatNlet, Nlet, NletDataset, SupervisedDataset
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig


class CombinedRandomSampler(ContrastiveSampler):
    def __init__(self, sampler_1: ContrastiveSampler, sampler_2: ContrastiveSampler = None) -> None:
        self.sampler_1 = sampler_1
        self.sampler_2 = sampler_2

    def __getitem__(self, idx: int) -> tuple[ContrastiveImage, int]:
        if self.sampler_2 is None:
            return self.sampler_1[idx]

        if idx < len(self.sampler_1):
            return (self.sampler_1[idx], 0)
        else:
            return (self.sampler_2[idx - len(self.sampler_1)], 1)

    def __len__(self) -> int:
        return len(self.sampler_1) + len(self.sampler_2) if self.sampler_2 is not None else len(self.sampler_1)

    @property
    def class_labels(self) -> list[gtypes.Label]:
        return self.sampler_1.class_labels + self.sampler_2.class_labels

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a different positive sample from the same class."""
        raise NotImplementedError

    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a negative sample from a different class."""
        raise NotImplementedError

    def negative_classes(self, sample: ContrastiveImage) -> list[Label]:
        """Return all possible negative labels for a sample"""
        raise NotImplementedError


class CombinedDataset(NletDataset):
    def __init__(
        self,
        base_dir: Path,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        transform: gtypes.TensorTransform,
        ssl_config: SSLConfig = None,
        dataset1_cls: Type[ContrastiveSampler] = SSLDataset,
        dataset2_cls: Type[ContrastiveSampler] = SupervisedDataset,
        **kwargs: Any,
    ) -> None:
        # base_dir -> is Path1:Path2
        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([self.get_transforms(), transform])
        self.partition = partition
        self.nlet_builder = nlet_builder
        path_1, path_2 = str(base_dir).split(":")
        if partition == "train":
            self.dataset_1 = dataset1_cls(
                base_dir=path_1,
                nlet_builder=nlet_builder,
                partition=partition,
                transform=transform,
                ssl_config=ssl_config,
                **kwargs,
            )
            self.dataset_2 = dataset2_cls(
                base_dir=path_2, nlet_builder=nlet_builder, partition=partition, transform=transform, **kwargs
            )
            self.contrastive_sampler = self.create_contrastive_sampler(base_dir)
        else:
            self.dataset_1 = dataset1_cls(
                base_dir=path_1,
                nlet_builder=nlet_builder,
                partition=partition,
                transform=transform,
                ssl_config=ssl_config,
                **kwargs,
            )
            self.contrastive_sampler = self.create_contrastive_sampler(base_dir)

    def __len__(self) -> int:
        return len(self.dataset_1) + len(self.dataset_2) if self.partition == "train" else len(self.dataset_1)

    def create_contrastive_sampler(
        self, base_dir: Path, sampler_class: type = CombinedRandomSampler
    ) -> CombinedRandomSampler:
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
        assert sampler_class == CombinedRandomSampler
        if self.partition == "train":
            return CombinedRandomSampler(self.dataset_1.contrastive_sampler, self.dataset_2.contrastive_sampler)
        else:
            return CombinedRandomSampler(self.dataset_1.contrastive_sampler)

    def class_distribution(self) -> dict[gtypes.Label, int]:  # TODO
        return None

    @property
    def num_classes(self) -> int:
        return self.dataset_2.num_classes if self.partition == "train" else -1
