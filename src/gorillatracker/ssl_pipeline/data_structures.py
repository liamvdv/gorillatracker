"""
### Directed Bipartite Graph

TLDR: Graph where nodes are divided into two disjoint sets, 
and edges only connect nodes from different sets.

### Union Graph

TLDR: Clique which can be unioned and can have negative edges between other cliques

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

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Generic, Protocol, TypeVar


class Hashable(Protocol):
    def __hash__(self) -> int: ...


CT = TypeVar("CT")


class Comparable(Protocol):
    def __lt__(self: CT, other: CT) -> bool: ...


class HashableComparable(Hashable, Comparable): ...


T = TypeVar("T", bound=Hashable)
K = TypeVar("K", bound=HashableComparable)


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
        self.members = {i: {i} for i in vertices}

    def find(self, x: T) -> T:
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x: T, y: T) -> T:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return root_y
        if self.rank[root_x] > self.rank[root_y]:
            root_x, root_y = root_y, root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_y] += 1
        self.root[root_x] = root_y
        self.members[root_y] |= self.members.pop(root_x)
        return root_y

    def get_members(self, x: T) -> set[T]:
        return self.members[self.find(x)]


class EdgeType(Enum):
    POSITIVE = 1
    NEGATIVE = -1


class UnionGraph(Generic[T]):
    def __init__(self, vertices: list[T]) -> None:
        assert len(vertices) == len(set(vertices)), "Vertices must be unique."
        self.union_find = UnionFind(vertices)
        self.negative_edges = {v: set[T]() for v in vertices}

    def add_edge(self, u: T, v: T, edge_type: EdgeType) -> None:
        assert u != v, "Self loops are not allowed."
        if edge_type is EdgeType.POSITIVE:
            assert not self.has_negative_relationship(
                u, v
            ), "Cannot add positive relationship between negatively connected groups."
            self._add_positive_edge(u, v)
        elif edge_type is EdgeType.NEGATIVE:
            assert not self.has_positive_relationship(
                u, v
            ), "Cannot add negative relationship between positively connected groups."
            self._add_negative_edge(u, v)

    def has_negative_relationship(self, u: T, v: T) -> bool:
        return self.union_find.find(u) in self.negative_edges[self.union_find.find(v)]

    def has_positive_relationship(self, u: T, v: T) -> bool:
        return self.union_find.find(u) == self.union_find.find(v)

    def get_group(self, v: T) -> tuple[T, set[T]]:
        root_v = self.union_find.find(v)
        members = self.union_find.get_members(v)
        return root_v, members

    def get_adjacent_negative_groups(self, v: T) -> dict[T, set[T]]:
        negative_roots = self._get_negative_edges(v)
        return {negative_root: self.union_find.get_members(negative_root) for negative_root in negative_roots}

    def _get_negative_edges(self, v: T) -> set[T]:
        root_v = self.union_find.find(v)
        return self.negative_edges[root_v]

    def _add_positive_edge(self, u: T, v: T) -> None:
        # we are merging groups
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        if root_u == root_v:  # prevent poppping the same key
            return
        root = self.union_find.union(u, v)
        old_root = root_v if root == root_u else root_u
        # transfer negative edges from the old group root to the new group root
        old_root_negative_neighbors = self.negative_edges.pop(old_root)
        for neighbor in old_root_negative_neighbors:
            self.negative_edges[neighbor].remove(old_root)
            self.negative_edges[neighbor].add(root)
        self.negative_edges[root] |= old_root_negative_neighbors

    def _add_negative_edge(self, u: T, v: T) -> None:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        self.negative_edges[root_u].add(root_v)
        self.negative_edges[root_v].add(root_u)


class IndexedUnionGraph(UnionGraph[K]):
    """UnionGraph with reproducible group identifiers and order of verticies
    independent of the edge insertion order."""

    def __init__(self, vertices: list[K]) -> None:
        super().__init__(vertices)
        self.index = sorted(vertices)
        assert all(
            self.index[i] < self.index[i + 1] for i in range(len(self.index) - 1)
        ), "Verticies must have an unique order"

    def get_group(self, v: K) -> tuple[K, set[K]]:
        members = self.union_find.get_members(v)
        return min(members), members

    def get_adjacent_negative_groups(self, v: K) -> dict[K, set[K]]:
        negative_roots = self._get_negative_edges(v)
        return {
            min(members): members
            for negative_root in negative_roots
            if (members := self.union_find.get_members(negative_root))
        }

    def __getitem__(self, key: int) -> K:
        return self.index[key]

    def __len__(self) -> int:
        return len(self.index)
