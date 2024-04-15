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
    """Graph where nodes are divided into two disjoint sets,
    and edges only connect nodes from different sets."""

    def __init__(self, left_nodes: list[T], right_nodes: list[T]) -> None:
        self.left_nodes = set(left_nodes)
        self.right_nodes = set(right_nodes)
        self.forward_edges: defaultdict[T, set[T]] = defaultdict(set)
        self.reverse_edges: defaultdict[T, set[T]] = defaultdict(set)

    def add_edge(self, left: T, right: T) -> None:
        assert left not in self.right_nodes
        assert right not in self.left_nodes
        assert left not in self.forward_edges[right]
        assert right not in self.forward_edges[left]

        self.left_nodes.add(left)
        self.right_nodes.add(right)

        self.forward_edges[left].add(right)
        self.reverse_edges[right].add(left)

    def bijective_relationships(self) -> set[tuple[T, T]]:
        bijective_pairs: set[tuple[T, T]] = set()

        for left, rights in self.forward_edges.items():
            if len(rights) == 1:
                right = next(iter(rights))
                if len(self.reverse_edges[right]) == 1:
                    assert next(iter(self.reverse_edges[right])) == left
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


class CliqueRelation(Enum):
    CONNECT = 1
    SEPARATE = -1


class CliqueGraph(Generic[T]):
    """A graph consisting of cliques, allowing operations to connect two cliques
    or establish a clear separation between them.

    This graph supports:
    - Merging cliques through connection relationships.
    - Defining explicit separations that prevent cliques from merging."""

    def __init__(self, vertices: list[T]) -> None:
        assert len(vertices) == len(set(vertices)), "Vertices must be unique."
        self.union_find = UnionFind(vertices)
        self.negative_connections = {v: set[T]() for v in vertices}

    def add_relationship(self, u: T, v: T, relation: CliqueRelation) -> None:
        assert u != v, "Self loops are not allowed."
        if relation is CliqueRelation.CONNECT:
            assert not self.is_separated(u, v), "Cannot add a positive edge between negatively connected cliques"
            self._add_connection(u, v)
        elif relation is CliqueRelation.SEPARATE:
            assert not self.is_connected(u, v), "Cannot add a negative edge in a clique"
            self._add_separation(u, v)

    def is_separated(self, u: T, v: T) -> bool:
        return self.union_find.find(u) in self.negative_connections[self.union_find.find(v)]

    def is_connected(self, u: T, v: T) -> bool:
        return self.union_find.find(u) == self.union_find.find(v)

    def get_clique(self, v: T) -> tuple[T, set[T]]:
        root_v = self.union_find.find(v)
        members = self.union_find.get_members(v)
        return root_v, members

    def get_adjacent_negative_cliques(self, v: T) -> dict[T, set[T]]:
        negative_roots = self._get_separations(v)
        return {negative_root: self.union_find.get_members(negative_root) for negative_root in negative_roots}

    def _get_separations(self, v: T) -> set[T]:
        root_v = self.union_find.find(v)
        return self.negative_connections[root_v]

    def _add_connection(self, u: T, v: T) -> None:
        # we are merging cliques
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        if root_u == root_v:  # prevent poppping the same key
            return
        root = self.union_find.union(u, v)
        old_root = root_v if root == root_u else root_u
        # transfer negative edges from the old clique root to the new clique root
        old_root_negative_neighbors = self.negative_connections.pop(old_root)
        for neighbor in old_root_negative_neighbors:
            self.negative_connections[neighbor].remove(old_root)
            self.negative_connections[neighbor].add(root)
        self.negative_connections[root] |= old_root_negative_neighbors

    def _add_separation(self, u: T, v: T) -> None:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        self.negative_connections[root_u].add(root_v)
        self.negative_connections[root_v].add(root_u)


class IndexedCliqueGraph(CliqueGraph[K]):
    """CliqueGraph with reproducible clique identifiers and order of verticies
    independent of the edge insertion order."""

    def __init__(self, vertices: list[K]) -> None:
        super().__init__(vertices)
        self.vertices = sorted(vertices)
        assert all(
            self.vertices[i] < self.vertices[i + 1] for i in range(len(self.vertices) - 1)
        ), "Verticies must have an unique order"

    def get_clique(self, v: K) -> tuple[K, set[K]]:
        members = self.union_find.get_members(v)
        return min(members), members

    def get_adjacent_negative_cliques(self, v: K) -> dict[K, set[K]]:
        negative_roots = self._get_separations(v)
        return {
            min(members): members
            for negative_root in negative_roots
            if (members := self.union_find.get_members(negative_root))
        }

    def __getitem__(self, key: int) -> K:
        return self.vertices[key]

    def __len__(self) -> int:
        return len(self.vertices)
