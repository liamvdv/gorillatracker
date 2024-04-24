from __future__ import annotations

from typing import Callable, Literal

from torch.utils.data import Dataset

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.contrastive_sampler import ContrastiveImage, ContrastiveSampler
from gorillatracker.type_helper import Nlet

FlatNlet = tuple[ContrastiveImage, ...]


# TODO(memben): add chache for val, test
class SSLDataset(Dataset[Nlet]):
    def __init__(
        self,
        contrastive_sampler: ContrastiveSampler,
        nlet_builder: Callable[[int, ContrastiveSampler], FlatNlet],
        partition: Literal["train", "val", "test"],
        transform: gtypes.Transform,
    ):
        self.contrastive_sampler = contrastive_sampler
        self.nlet_builder = nlet_builder
        self.transform = transform
        self.partition = partition

    def __len__(self) -> int:
        return len(self.contrastive_sampler)

    def __getitem__(self, idx: int) -> Nlet:
        flat_nlet = self.nlet_builder(idx, self.contrastive_sampler)
        return self.stack_flat_nlet(flat_nlet)

    def stack_flat_nlet(self, flat_nlet: FlatNlet) -> Nlet:
        ids = tuple(img.image_path for img in flat_nlet)
        labels = tuple(img.class_label for img in flat_nlet)
        values = tuple(self.transform(img.image) for img in flat_nlet)
        return Nlet(ids, values, labels)


def build_triplet(
    idx: int, contrastive_sampler: ContrastiveSampler
) -> tuple[ContrastiveImage, ContrastiveImage, ContrastiveImage]:
    anchor_positive = contrastive_sampler[idx]
    positive = contrastive_sampler.positive(anchor_positive)
    negative = contrastive_sampler.negative(anchor_positive)
    return anchor_positive, positive, negative


def build_quadruplet(
    idx: int, contrastive_sampler: ContrastiveSampler
) -> tuple[ContrastiveImage, ContrastiveImage, ContrastiveImage, ContrastiveImage]:
    anchor_positive = contrastive_sampler[idx]
    positive = contrastive_sampler.positive(anchor_positive)
    anchor_negative = contrastive_sampler.negative(anchor_positive)
    negative = contrastive_sampler.positive(anchor_negative)
    return anchor_positive, positive, anchor_negative, negative
