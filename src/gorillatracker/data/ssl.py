from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import torch
from PIL.Image import Image
from torchvision import transforms

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import ContrastiveSampler
from gorillatracker.data.nlet import FlatNlet, NletDataset
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig


class SSLDataset(NletDataset):
    def __init__(
        self,
        base_dir: Path,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        transform: gtypes.TensorTransform,
        ssl_config: SSLConfig,
        **kwargs: Any,
    ):
        self.ssl_config = ssl_config
        self.nlet_builder = nlet_builder
        self.transform: Callable[[Image], torch.Tensor] = transforms.Compose([self.get_transforms(), transform])
        self.partition: Literal["train", "val", "test"] = partition
        self.contrastive_sampler = self.create_contrastive_sampler(base_dir)

    @property
    def num_classes(self) -> int:
        raise ValueError("num classes unknown for SSLDataset")

    @property
    def class_distribution(self) -> dict[gtypes.Label, int]:
        raise ValueError("class distribution unknown for SSLDataset")

    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveSampler:
        return self.ssl_config.get_contrastive_sampler(base_dir, self.partition)
