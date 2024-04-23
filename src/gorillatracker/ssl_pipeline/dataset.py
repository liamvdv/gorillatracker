from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph
from gorillatracker.type_helper import FlatNlet, ImageLabel


@dataclass(frozen=True)  # TODO(memben): Slots?
class LazyImage:
    id: int
    img_path: Path

    # NOTE(memben): for mypy, does not recognize order=True
    def __lt__(self, other: LazyImage) -> bool:
        return self.id < other.id

    def load(self, transform: gtypes.Transform, label: int) -> ImageLabel:
        img = Image.open(self.img_path)
        img = transform(img)
        return str(self.img_path), img, label

    def get_label(self, graph: IndexedCliqueGraph[LazyImage]) -> int:
        return graph.get_clique_representative(self).id


class SSLDataset(Dataset[FlatNlet]):
    def __init__(
        self,
        graph: IndexedCliqueGraph[LazyImage],
        nlet_builder: Callable[[int, IndexedCliqueGraph[LazyImage]], tuple[LazyImage, ...]],
        partition: Literal["train", "val", "test"],
        transform: gtypes.Transform,
    ):
        self.graph = graph
        self.nlet_builder = nlet_builder
        self.transform = transform
        self.partition = partition
        # self.cache: Optional[defaultdict[int, Nlet]] = None
        # if self.partition == "val" or self.partition == "test":
        #     self.cache = defaultdict(None)

    def __len__(self) -> int:
        return len(self.graph)

    def __getitem__(self, idx: int) -> FlatNlet:
        # if self.cache is not None and idx in self.cache:
        #     return self.cache[idx]
        nlet = self.nlet_builder(idx, self.graph)
        return tuple(img.load(self.transform, img.get_label(self.graph)) for img in nlet)  # TODO(memben)


def build_triplet(idx: int, graph: IndexedCliqueGraph[LazyImage]) -> tuple[LazyImage, LazyImage, LazyImage]:
    anchor = graph[idx]
    positive = graph.get_random_clique_member(anchor, exclude=[anchor])
    negative = graph.get_random_adjacent_clique_member(anchor)
    return anchor, positive, negative


def build_quadlet(idx: int, graph: IndexedCliqueGraph[LazyImage]) -> tuple[LazyImage, LazyImage, LazyImage, LazyImage]:
    anchor_positive = graph[idx]
    positive = graph.get_random_clique_member(anchor_positive, exclude=[anchor_positive])
    anchor_negative = graph.get_random_adjacent_clique_member(anchor_positive)
    negative = graph.get_random_clique_member(anchor_negative, exclude=[anchor_negative])
    return anchor_positive, positive, anchor_negative, negative
