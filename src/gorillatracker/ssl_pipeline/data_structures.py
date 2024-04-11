"""
### Directed Bipartite Graph

TLDR: Graph where nodes are divided into two disjoint sets, 
and edges only connect nodes from different sets.

### Union Graph

TLDR: Reduce the number of labelling by utilizing the graph's structure to infer
as much as possible about the relationships between nodes before adding new labels.

We're dealing with a graph where nodes represent entities and edges represent relationships
between these entities. Edges can represent a positive or negative relationship

Positive Relationship: If two nodes have a positive edge between them,
they are considered to be in the same group. This relationship is transitive within the group,
meaning if node A is connected to node B, and node B is connected to node C,
all with positive edges, then A, B, and C are in the same group.

Negative Relationship: If two nodes have a negative edge between them,
they are considered to be in different groups. This relationship is not transitive,
meaning if node A is connected to node B with a negative edge, and node B is connected to node C with a negative edge,
it does not imply that A and C are negatively connected.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar


class Hashable(Protocol):
    def __hash__(self) -> int: ...


T = TypeVar("T", bound=Hashable)


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


class UnionFind(Generic[T]):
    def __init__(self, vertices: list[T]) -> None:
        self.root = {i: i for i in vertices}
        self.rank = {i: 1 for i in vertices}

    def find(self, x: T) -> T:
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x: T, y: T) -> None:
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1


class EdgeType(Enum):
    POSITIVE = 1
    NEGATIVE = -1


@dataclass(frozen=True)
class Edge:
    vertex: int
    type: EdgeType


class UnionGraph(Generic[T]):
    """A graph that keeps track of the relationships between groups of vertices."""

    def __init__(self, vertices: list[T]):
        self.union_find = UnionFind[T](vertices)
        self.groups = {i: {i} for i in vertices}
        self.negative_relations: dict[T, set[T]] = {i: set[T]() for i in vertices}

    def add_edge(self, u: T, v: T, type: EdgeType) -> None:
        assert u != v, "Invalid edge"
        if type == EdgeType.POSITIVE:
            assert not self.has_group_negative_edge(u, v)
            root_u = self.union_find.find(u)
            root_v = self.union_find.find(v)
            self.union_find.union(u, v)
            self._merge_groups(root_u, root_v)
        elif type == EdgeType.NEGATIVE:
            assert not self.is_same_group(u, v)
            self._add_negative_edge(u, v)

    def get_group(self, vertex: T) -> set[T]:
        return self.groups[self.union_find.find(vertex)]

    def _merge_groups(self, root_u: T, root_v: T) -> None:
        root = self.union_find.find(root_u)
        self.groups[root] = self.groups[root_u] | self.groups[root_v]
        if root_u != root:
            del self.groups[root_u]
        if root_v != root:
            del self.groups[root_v]

    def _add_negative_edge(self, u: T, v: T) -> None:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        self.negative_relations[root_u].add(root_v)
        self.negative_relations[root_v].add(root_u)

    def has_group_negative_edge(self, u: T, v: T) -> bool:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        return root_v in self.negative_relations[root_u] or root_u in self.negative_relations[root_v]

    def is_same_group(self, u: T, v: T) -> bool:
        return self.union_find.find(u) == self.union_find.find(v)
