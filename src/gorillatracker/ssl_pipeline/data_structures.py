from __future__ import annotations

import random
from collections import defaultdict
from itertools import chain
from typing import Generic, Protocol, TypeVar

CT = TypeVar("CT")


class Comparable(Protocol):
    def __lt__(self: CT, other: CT, /) -> bool: ...


T = TypeVar("T")


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
        assert len(vertices) == len(set(vertices)), "Vertices must be unique."
        self.root = {i: i for i in vertices}
        self.rank = {i: 1 for i in vertices}
        self.members = {i: [i] for i in vertices}

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
        self.members[root_y].extend(self.members.pop(root_x))
        return root_y

    def get_members(self, x: T) -> list[T]:
        return self.members[self.find(x)]

    def delete_set(self, root: T) -> None:
        members_to_delete = self.members.pop(root)
        for member in members_to_delete:
            self.root.pop(member)
            self.rank.pop(member)


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

    def get_clique(self, v: T) -> list[T]:
        return self.union_find.get_members(v)

    def get_adjacent_cliques(self, v: T) -> dict[T, list[T]]:
        adjacent_clique_roots = self._get_adjacent_clique_roots(v)
        return {r: self.get_clique(r) for r in adjacent_clique_roots}

    def get_random_clique_member(self, v: T, exclude: list[T] = []) -> T:
        clique = self.get_clique(v)
        return random.choice([m for m in clique if m not in exclude])

    def get_random_adjacent_clique(self, v: T) -> T:
        adjacent_clique_roots = self._get_adjacent_clique_roots(v)
        return random.choice(list(adjacent_clique_roots))

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

    def _get_adjacent_clique_roots(self, v: T) -> set[T]:
        root_v = self._find_root(v)
        return self.cut_edges[root_v]


K = TypeVar("K", bound=Comparable)


class IndexedCliqueGraph(CliqueGraph[K]):
    """CliqueGraph with reproducible clique roots named representatives and order of verticies
    independent of the edge insertion order."""

    def __init__(self, vertices: list[K]) -> None:
        super().__init__(vertices)
        self.vertices = sorted(vertices)
        assert all(
            self.vertices[i] < self.vertices[i + 1] for i in range(len(self.vertices) - 1)
        ), "Verticies must have an unique order"

    def get_clique_representative(self, v: K) -> K:
        return min(self.get_clique(v))

    # override
    def get_adjacent_cliques(self, v: K) -> dict[K, list[K]]:
        adjacent_clique_roots = self._get_adjacent_clique_roots(v)
        adjacent_cliques = {}
        for r in adjacent_clique_roots:
            clique = self.get_clique(r)
            adjacent_cliques[min(clique)] = clique
        return adjacent_cliques

    def __getitem__(self, key: int) -> K:
        return self.vertices[key]

    def __len__(self) -> int:
        return len(self.vertices)


P = TypeVar("P", bound=Comparable)


class MultiLayerCliqueGraph(IndexedCliqueGraph[K]):
    """Indexed Clique Graph supporting multiple layers of cut edges.
    This allows for hierarchical or multi-layered connections between cliques.
    A clique in a parent layer does not guarantee that the children are connected
    but a partition in a parent layer ensures a partition in the child layers."""

    def __init__(self, vertices: list[K], parent: CliqueGraph[P], parent_edges: dict[K, P | None]) -> None:
        """
        Args:
            vertices: List of vertices in the current layer
            parent: Parent CliqueGraph
            parent_edges: Mapping from the current layer vertices to the parent layer vertices
        """
        super().__init__(vertices)
        self.parent = parent
        self.parent_edges = parent_edges
        self.inverse_parent_edges: defaultdict[P, set[K]] = defaultdict(set)
        for child_r, parent_r in parent_edges.items():
            if parent_r is not None:
                self.inverse_parent_edges[parent_r].add(child_r)

    # override
    def merge(self, u: K, v: K) -> None:
        u_root, v_root = self._find_root(u), self._find_root(v)
        super().merge(u, v)
        parent_u, parent_v = self.parent_edges.pop(u_root, None), self.parent_edges.pop(v_root, None)
        if parent_u is not None and parent_v is not None:
            assert parent_u == parent_v, "MultiParent merge not supported"
        new_root = self._find_root(u)
        self.parent_edges[new_root] = parent_u or parent_v

    # override
    def is_partitioned(self, u: K, v: K) -> bool:
        self_p = super().is_partitioned(u, v)
        parent_p = self._is_parent_partitioned(u, v)
        return self_p or parent_p

    def _is_parent_partitioned(self, u: K, v: K) -> bool:
        root_u, root_v = self._find_root(u), self._find_root(v)
        parent_u, parent_v = self.parent_edges.get(root_u), self.parent_edges.get(root_v)
        if parent_u is None or parent_v is None:
            return False
        return self.parent.is_partitioned(parent_u, parent_v)

    # override
    def get_adjacent_cliques(self, v: K) -> dict[K, list[K]]:
        return super().get_adjacent_cliques(v) | self._get_adjacent_cliques_via_parent(v)

    def _get_adjacent_cliques_via_parent(self, v: K) -> dict[K, list[K]]:
        adjacent_clique_representatives = self._get_adjacent_clique_roots_via_parent(v)
        return {r: self.get_clique(r) for r in adjacent_clique_representatives}

    # override
    def _get_adjacent_clique_roots(self, v: K) -> set[K]:
        return super()._get_adjacent_clique_roots(v) | self._get_adjacent_clique_roots_via_parent(v)

    def _get_adjacent_clique_roots_via_parent(self, v: K) -> set[K]:
        # 1. We go one layer up and collect all adjacent cliques in the parent layer => parent_adjacent_cliques
        #       This might be done recursively
        # 2. We get all nodes in the parent layer that are adjacent to the parent clique => adjacent_clique_parents
        # 3. We get all children of the adjacent parent cliques elements => adjacent_clique_representatives
        root_v = self._find_root(v)
        parent_v = self.parent_edges.get(root_v)
        if parent_v is None:
            return set()
        parent_adjacent_cliques = self.parent.get_adjacent_cliques(parent_v)
        adjacent_clique_parents = chain.from_iterable(parent_adjacent_cliques.values())
        adjacent_clique_representatives = list(
            chain.from_iterable([self.inverse_parent_edges[p] for p in adjacent_clique_parents])
        )

        assert len(adjacent_clique_representatives) == len(
            set(adjacent_clique_representatives)
        ), "Must be unique. Logic Error."
        return set(adjacent_clique_representatives)

    def prune_cliques_without_neighbors(self) -> None:
        deleted_vertices = set()
        roots = self.union_find.members.keys()
        for root in list(roots):
            if len(self.get_adjacent_cliques(root)) == 0:
                deleted_vertices.update(self.get_clique(root))
                self.union_find.delete_set(root)
        self.vertices = list(set(self.vertices) - deleted_vertices)
