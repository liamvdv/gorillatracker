from __future__ import annotations

from collections import defaultdict
from typing import Generic, Protocol, TypeVar

CT = TypeVar("CT")


class Comparable(Protocol):
    def __lt__(self: CT, other: CT) -> bool: ...


T = TypeVar("T")
K = TypeVar("K", bound=Comparable)


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


class CliqueGraph(Generic[T]):
    """A graph consisting of cliques, allowing operations to merge two cliques
    or establish a clear separation between them."""

    def __init__(self, vertices: list[T]) -> None:
        assert len(vertices) == len(set(vertices)), "Vertices must be unique."
        self.union_find = UnionFind(vertices)
        # NOTE(memben): the key and values are always the root of a set in union find
        self.cut_edges = {v: set[T]() for v in vertices}

    def is_partitioned(self, u: T, v: T) -> bool:
        root_u, root_v = self._find_root(u), self._find_root(v)
        return root_u in self.cut_edges[root_v]

    def is_connected(self, u: T, v: T) -> bool:
        return self._find_root(u) == self._find_root(v)

    def get_clique(self, v: T) -> set[T]:
        return self.union_find.get_members(v)

    def get_adjacent_cliques(self, v: T) -> dict[T, set[T]]:
        adjacent_clique_roots = self._get_adjacent_partitions(v)
        return {r: self.get_clique(r) for r in adjacent_clique_roots}

    def merge(self, u: T, v: T) -> None:
        assert u != v, "Self loops are not allowed."
        assert not self.is_partitioned(u, v), "Cannot merge partitioned cliques"
        root_u, root_v = self._find_root(u), self._find_root(v)
        if root_u == root_v:
            return
        root = self.union_find.union(u, v)
        old_root = root_v if root == root_u else root_u
        old_cut_edges = self.cut_edges.pop(old_root)
        for root_p in old_cut_edges:
            self.cut_edges[root_p].remove(old_root)
            self.cut_edges[root_p].add(root)
        self.cut_edges[root] |= old_cut_edges

    def partition(self, u: T, v: T) -> None:
        assert u != v, "Self loops are not allowed."
        assert not self.is_connected(u, v), "Cannot partition a clique"
        root_u, root_v = self._find_root(u), self._find_root(v)
        self.cut_edges[root_u].add(root_v)
        self.cut_edges[root_v].add(root_u)

    def _find_root(self, v: T) -> T:
        return self.union_find.find(v)

    def _get_adjacent_partitions(self, v: T) -> set[T]:
        root_v = self._find_root(v)
        return self.cut_edges[root_v]


class IndexedCliqueGraph(CliqueGraph[K]):
    """CliqueGraph with reproducible clique identifiers and order of verticies
    independent of the edge insertion order."""

    def __init__(self, vertices: list[K]) -> None:
        super().__init__(vertices)
        self.vertices = sorted(vertices)
        assert all(
            self.vertices[i] < self.vertices[i + 1] for i in range(len(self.vertices) - 1)
        ), "Verticies must have an unique order"

    def get_clique_representative(self, v: K) -> K:
        return min(self.get_clique(v))

    def get_adjacent_cliques(self, v: K) -> dict[K, set[K]]:
        adjacent_clique_roots = self._get_adjacent_partitions(v)
        adjacent_cliques = {}
        for r in adjacent_clique_roots:
            clique = self.get_clique(r)
            adjacent_cliques[min(clique)] = clique
        return adjacent_cliques

    def __getitem__(self, key: int) -> K:
        return self.vertices[key]

    def __len__(self) -> int:
        return len(self.vertices)
