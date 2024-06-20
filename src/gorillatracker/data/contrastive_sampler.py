import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph
from gorillatracker.type_helper import Id, Label

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True, slots=True)  # type: ignore
class ContrastiveImage:
    id: Id
    image_path: Path
    class_label: Label

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)


def group_contrastive_images(
    contrastive_images: list[ContrastiveImage],
) -> defaultdict[gtypes.Label, list[ContrastiveImage]]:
    classes: defaultdict[gtypes.Label, list[ContrastiveImage]] = defaultdict(list)
    for image in contrastive_images:
        classes[image.class_label].append(image)
    return classes


class ContrastiveSampler(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> ContrastiveImage:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[ContrastiveImage]:
        """Provides an iterator over the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    @property
    @abstractmethod
    def class_labels(self) -> list[gtypes.Label]:
        pass

    @abstractmethod
    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a different positive sample from the same class."""
        pass

    @abstractmethod
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a negative sample from a different class."""
        pass

    @abstractmethod
    def negative_classes(self, sample: ContrastiveImage) -> list[Label]:
        """Return all possible negative labels for a sample"""
        pass

    # HACK(memben): ...
    def find_any_image(self, label: Label) -> ContrastiveImage:
        for image in self:
            if image.class_label == label:
                return image
        raise ValueError(f"No image found for label {label}")


class ContrastiveClassSampler(ContrastiveSampler):
    """ContrastiveSampler that samples from a set of classes. Negatives are drawn from a uniformly sampled negative class"""

    def __init__(self, classes: dict[gtypes.Label, list[ContrastiveImage]]) -> None:
        self.classes = classes
        self.samples = [sample for samples in classes.values() for sample in samples]
        self.sample_to_class = {sample: label for label, samples in classes.items() for sample in samples}

        # assert all([len(samples) > 1 for samples in classes.values()]), "Classes must have at least two samples" # TODO(memben)
        for label, samples in classes.items():
            if len(samples) < 2:
                logger.warning(f"Class {label} has less than two samples (samples: {len(samples)}).")

        assert len(self.samples) == len(set(self.samples)), "Samples must be unique"

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def class_labels(self) -> list[gtypes.Label]:
        return list(self.classes.keys())

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        positive_class = self.sample_to_class[sample]
        if len(self.classes[positive_class]) == 1:
            # logger.warning(f"Only one sample in class {positive_class}. Returning same sample as positive.")
            return sample
        positives = [s for s in self.classes[positive_class] if s != sample]
        return random.choice(positives)

    # NOTE(memben): First samples a negative class to ensure a more balanced distribution of negatives,
    # independent of the number of samples per class
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Different class is sampled uniformly at random and a random sample from that class is returned"""
        negative_class = random.choice(self.negative_classes(sample))
        negatives = self.classes[negative_class]
        return random.choice(negatives)

    def negative_classes(self, sample: ContrastiveImage) -> list[Label]:
        positive_class = self.sample_to_class[sample]
        negative_classes = [c for c in self.class_labels if c != positive_class]
        return negative_classes


class CliqueGraphSampler(ContrastiveSampler):
    def __init__(self, graph: IndexedCliqueGraph[ContrastiveImage]):
        self.graph = graph

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.graph[idx]

    def __len__(self) -> int:
        return len(self.graph)

    @property
    def class_labels(self) -> list[gtypes.Label]:
        raise NotImplementedError("No logic yet implemented")

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        return self.graph.get_random_clique_member(sample, exclude=[sample])

    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        random_adjacent_clique = self.graph.get_random_adjacent_clique(sample)
        return self.graph.get_random_clique_member(random_adjacent_clique)

    # TODO(memben): if this becomes a bottleneck, consider only retrieving the roots
    def negative_classes(self, sample: ContrastiveImage) -> list[Label]:
        adjacent_cliques = self.graph.get_adjacent_cliques(sample)
        return [root.class_label for root in adjacent_cliques.keys()]
