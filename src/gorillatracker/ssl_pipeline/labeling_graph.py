"""
TLDR: Reduce the number of labeling by utilizing the graph's structure to infer 
as much as possible about the relationships between nodes before adding new labels.


We're dealing with a graph where nodes represent entities and edges represent relationships 
between these entities. Edges can have a value of either 1 or -1,
symbolizing positive and negative relationships, respectively.

Positive Relationship (1): If two nodes are connected by an edge with a value of 1, 
they are considered to be in the same group. This relationship is transitive within the group, 
meaning if node A is connected to node B, and node B is connected to node C, 
all with positive edges, then A, B, and C are in the same group.

Negative Relationship (-1): If two nodes are connected by an edge with a value of -1, 
they are considered to be in different groups. However, 
this negative relationship does not transitively apply to connections between other groups. 
For instance, if group A is negatively connected to group B, 
and group B is negatively connected to group C, 
it does not imply that group A and group C are negatively connected.

The goal is to determine whether a new connection between two nodes should be labeled with a 1 or -1. 
However, we aim to avoid labeling connections that are redundant,
 meaning their relationship status (same group or different groups) 
 can already be inferred from the existing graph structure.

Criteria for Labeling:

Label with 1: If it's not already established that the two nodes belong to the same group through a series of positive connections.
Label with -1: If the two nodes are known to be in different groups based on the existing structure, specifically if there's any direct or indirect negative connection that differentiates their groups.

Objective: Minimize unnecessary labeling by utilizing the graph's structure to infer 
as much as possible about the relationships between nodes before adding new labels
"""

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class UnionFind:
    def __init__(self, size: int):
        self.root = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, x: int) -> int:
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x: int, y: int) -> None:
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


class Relationship(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    CONTROL_POSITIVE = 2
    CONTROL_NEGATIVE = -2


@dataclass(frozen=True)
class Edge:
    vertex: int
    relationship: Relationship


class Graph:
    def __init__(self, vertices: int):
        self._graph: list[set[Edge]] = [set() for _ in range(vertices)]

    def add_edge(self, u: int, v: int, relationship: Relationship) -> None:
        assert u != v, "Invalid edge"
        assert not any(edge.vertex == v for edge in self._graph[u]), "Edge already exists"
        assert not any(edge.vertex == u for edge in self._graph[v]), "Edge already exists"
        self._graph[u].add(Edge(v, relationship))
        self._graph[v].add(Edge(u, relationship))

    def remove_edge(self, u: int, v: int) -> None:
        self._graph[u] = {edge for edge in self._graph[u] if edge.vertex != v}
        self._graph[v] = {edge for edge in self._graph[v] if edge.vertex != u}

    def __getitem__(self, vertex: int) -> set[Edge]:
        return self._graph[vertex]

    def __iter__(self) -> Iterator[set[Edge]]:
        return iter(self._graph)

    def __len__(self) -> int:
        return len(self._graph)


class UnionGraph:
    """A graph that keeps track of the relationships between groups of vertices"""

    def __init__(self, vertices: int):
        self.union_find = UnionFind(vertices)
        self.group = {i: {i} for i in range(vertices)}
        self.negative_relations: dict[int, set[int]] = {i: set() for i in range(vertices)}

    def add_edge(self, u: int, v: int, relationship: Relationship) -> None:
        assert u != v, "Invalid edge"
        if relationship == Relationship.POSITIVE:
            assert not self.has_group_negative_edge(u, v)
            root_u = self.union_find.find(u)
            root_v = self.union_find.find(v)
            self.union_find.union(u, v)
            self._merge_groups(root_u, root_v)
        elif relationship == Relationship.NEGATIVE:
            assert not self.is_same_group(u, v)
            self._add_negative_edge(u, v)

    def _merge_groups(self, root_u: int, root_v: int) -> None:
        root = self.union_find.find(root_u)
        self.group[root] = self.group[root_u] | self.group[root_v]
        if root_u != root:
            del self.group[root_u]
        if root_v != root:
            del self.group[root_v]

    def _add_negative_edge(self, u: int, v: int) -> None:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        self.negative_relations[root_u].add(root_v)
        self.negative_relations[root_v].add(root_u)

    def has_group_negative_edge(self, u: int, v: int) -> bool:
        root_u, root_v = self.union_find.find(u), self.union_find.find(v)
        return root_v in self.negative_relations[root_u] or root_u in self.negative_relations[root_v]

    def is_same_group(self, u: int, v: int) -> bool:
        return self.union_find.find(u) == self.union_find.find(v)


def transform_to_union_graph(label_graph: Graph) -> UnionGraph:
    union_graph = UnionGraph(len(label_graph))
    for vertex, edges in enumerate(label_graph):
        for edge in edges:
            union_graph.add_edge(vertex, edge.vertex, edge.relationship)
    return union_graph


class LabelTask:
    def __init__(self, label_graph: Graph, label_queue: deque[tuple[int, int]], control_probability: float):
        """
        Args:
        label_graph: Graph - the graph representing the relationships between entities
        label_queue: deque[tuple[int, int]] - a queue of connections to be labeled
        control_probability: float - the probability of a control labeling task **not** being skipped
        """
        assert all((b, a) not in label_queue for a, b in label_queue), "Queue contains duplicate edges"
        self.label_graph = label_graph
        self.label_queue = label_queue
        self.union_graph = transform_to_union_graph(label_graph)
        self.control_probability = control_probability
        self.history: list[tuple[int, int]] = []
        self.passed: deque[tuple[int, int]] = deque()

    def is_edge_redudant(self, id_1: int, id_2: int) -> bool:
        if self.union_graph.is_same_group(id_1, id_2):
            return True
        if self.union_graph.has_group_negative_edge(id_1, id_2):
            return True
        return False

    def label(self, id_1: int, id_2: int, relationship: Relationship) -> None:
        self.label_graph.add_edge(id_1, id_2, relationship)
        self.union_graph.add_edge(id_1, id_2, relationship)
        self.history.append((id_1, id_2))

    def undo(self) -> None:
        id_1, id_2 = self.history.pop()
        self.label_graph.remove_edge(id_1, id_2)
        self.union_graph = transform_to_union_graph(self.label_graph)
        self.reevaluate_passed()

    def reevaluate_passed(self) -> None:
        for id_1, id_2 in reversed(self.passed):
            if not self.is_edge_redudant(id_1, id_2) and (id_1, id_2) not in self.label_queue:
                self.label_queue.appendleft((id_1, id_2))

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return self

    def __next__(self) -> tuple[int, int]:
        if not self.label_queue:
            raise StopIteration
        id_1, id_2 = self.label_queue.popleft()
        self.passed.append((id_1, id_2))
        if self.is_edge_redudant(id_1, id_2):
            if random.random() > self.control_probability:
                return self.__next__()
        return (id_1, id_2)
