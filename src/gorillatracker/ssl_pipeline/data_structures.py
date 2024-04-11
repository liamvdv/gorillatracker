from collections import defaultdict
from typing import Generic, TypeVar

T = TypeVar("T")


class DirectedBipartiteGraph(Generic[T]):
    def __init__(self, left_nodes: list[T], right_nodes: list[T]) -> None:
        self.left_nodes = set(left_nodes)
        self.right_nodes = set(right_nodes)
        self.edges: defaultdict[T, set[T]] = defaultdict(set)
        self.inverse_edges: defaultdict[T, set[T]] = defaultdict(set)

    def add_edge(self, left: T, right: T) -> None:
        assert left not in self.right_nodes
        assert right not in self.left_nodes
        assert left not in self.edges[right]
        assert right not in self.edges[left]

        self.left_nodes.add(left)
        self.right_nodes.add(right)

        self.edges[left].add(right)
        self.inverse_edges[right].add(left)

    def bijective_relationships(self) -> set[tuple[T, T]]:
        bijective_pairs: set[tuple[T, T]] = set()

        for left, rights in self.edges.items():
            if len(rights) == 1:
                right = next(iter(rights))
                if len(self.inverse_edges[right]) == 1:
                    assert next(iter(self.inverse_edges[right])) == left
                    bijective_pairs.add((left, right))

        return bijective_pairs
