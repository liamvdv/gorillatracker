from dataclasses import dataclass
from abc import ABC, abstractmethod 

from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph

from PIL import Image

@dataclass(frozen=True, order=True)
class ContrastiveImage:
    id: int
    image_path: str
    class_label: int

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)


class ContrastiveSampler(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> ContrastiveImage:
        pass

    @abstractmethod
    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a different positive sample from the same class."""
        pass

    @abstractmethod
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a negative sample from a different class."""
        pass

class CliqueGraphSampler(ContrastiveSampler):
    def __init__(self, graph: IndexedCliqueGraph[ContrastiveImage]):
        self.graph = graph

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.graph[idx]

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        return self.graph.get_random_clique_member(sample)

    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        return self.graph.get_random_adjacent_clique_member(sample)
    
