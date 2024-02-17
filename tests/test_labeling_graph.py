from collections import deque

import pytest

from gorillatracker.ssl_pipeline.labeling_graph import (
    Graph,
    LabelTask,
    Relationship,
    UnionFind,
    UnionGraph,
    transform_to_union_graph,
)


@pytest.fixture
def setup_union_find() -> UnionFind:
    return UnionFind(10)


@pytest.fixture
def setup_graph() -> Graph:
    graph = Graph(5)
    graph.add_edge(0, 1, Relationship.POSITIVE)
    graph.add_edge(1, 2, Relationship.POSITIVE)
    graph.add_edge(2, 3, Relationship.NEGATIVE)
    graph.add_edge(3, 4, Relationship.NEGATIVE)
    return graph


@pytest.fixture
def setup_union_graph(setup_graph: Graph) -> UnionGraph:
    return transform_to_union_graph(setup_graph)


@pytest.fixture
def setup_label_task(setup_graph: Graph) -> LabelTask:
    label_queue: deque[tuple[int, int]] = deque([(0, 1), (1, 2), (0, 4)])
    return LabelTask(setup_graph, label_queue, 0.0)


def test_union_find_union_and_find(setup_union_find: UnionFind) -> None:
    uf = setup_union_find
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2), "UnionFind union and find operations failed."
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3), "UnionFind union and find operations failed."


def test_graph_add_and_check_edge(setup_graph: Graph) -> None:
    graph = setup_graph
    assert any(edge.vertex == 1 for edge in graph[0]), "Graph edge addition failed."


def test_union_graph_group_relationship(setup_union_graph: UnionGraph) -> None:
    u_graph = setup_union_graph
    assert u_graph.is_same_group(0, 1), "UnionGraph positive relationship check failed."
    assert not u_graph.has_group_negative_edge(0, 1), "UnionGraph positive relationship check failed."
    assert not u_graph.is_same_group(0, 3), "UnionGraph negative relationship check failed."
    assert u_graph.has_group_negative_edge(0, 3), "UnionGraph negative relationship check failed."

    assert not u_graph.is_same_group(0, 4), "UnionGraph negative relationship check failed."


def test_union_merge_groups(setup_union_graph: UnionGraph) -> None:
    u_graph = setup_union_graph
    assert not u_graph.is_same_group(0, 4), "UnionGraph negative relationship check failed."
    u_graph.add_edge(1, 4, Relationship.POSITIVE)
    assert u_graph.is_same_group(0, 4), "UnionGraph merge groups failed."


def test_graph_fails_invalid_edge(setup_graph: Graph) -> None:
    graph = setup_graph
    with pytest.raises(AssertionError):
        graph.add_edge(0, 0, Relationship.POSITIVE)
    with pytest.raises(AssertionError):
        graph.add_edge(0, 0, Relationship.NEGATIVE)
    with pytest.raises(AssertionError):
        graph.add_edge(1, 0, Relationship.POSITIVE)


def test_union_graph_fails_invalid_edge(setup_union_graph: UnionGraph) -> None:
    u_graph = setup_union_graph
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 0, Relationship.NEGATIVE)
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 1, Relationship.NEGATIVE)
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 3, Relationship.POSITIVE)


def test_adding_invalid_control_edges(setup_graph: Graph, setup_union_graph: UnionGraph) -> None:
    graph = setup_graph
    u_graph = setup_union_graph
    graph.add_edge(0, 2, Relationship.CONTROL_POSITIVE)
    graph.add_edge(0, 3, Relationship.CONTROL_NEGATIVE)

    u_graph.add_edge(0, 2, Relationship.CONTROL_POSITIVE)
    u_graph.add_edge(0, 3, Relationship.CONTROL_NEGATIVE)


def test_label_task_edge_redundancy(setup_label_task: LabelTask) -> None:
    task = setup_label_task
    assert task.is_edge_redudant(0, 1), "LabelTask redundancy check failed for positive relationship."
    assert not task.is_edge_redudant(0, 4), "LabelTask redundancy check failed for non-redundant edge."


def test_invalid_deque(setup_graph: Graph) -> None:
    label_queue: deque[tuple[int, int]] = deque([(0, 1), (1, 2), (1, 0)])
    with pytest.raises(AssertionError):
        LabelTask(setup_graph, label_queue, 0.0)


def test_queue_skipping(setup_graph: Graph) -> None:
    label_queue: deque[tuple[int, int]] = deque([(0, 1), (1, 2), (0, 4)])
    label_task = LabelTask(setup_graph, label_queue, 0.0)
    next_task = next(label_task)
    assert next_task == (0, 4), "LabelTask queue skipping failed."
    label_task.label(0, 4, Relationship.POSITIVE)
    assert label_task.union_graph.is_same_group(0, 4)


def test_label_task_undo() -> None:
    graph = Graph(5)
    graph.add_edge(0, 2, Relationship.POSITIVE)
    label_queue: deque[tuple[int, int]] = deque([(0, 1), (1, 2), (0, 4), (0, 3)])
    label_task = LabelTask(graph, label_queue, 0.0)

    next_task = next(label_task)
    assert next_task == (0, 1), "LabelTask queue skipping failed."
    label_task.label(0, 1, Relationship.POSITIVE)
    next_task = next(label_task)
    assert next_task == (0, 4), "LabelTask queue skipping failed."
    label_task.undo()
    next_task = next(label_task)
    assert next_task == (0, 1), "LabelTask queue undoing failed."
    label_task.label(0, 1, Relationship.NEGATIVE)
    next_task = next(label_task)
    assert next_task == (0, 4), "LabelTask queue skipping failed."
    label_task.label(0, 4, Relationship.POSITIVE)
    label_task.undo()
    next_task = next(label_task)
    assert next_task == (0, 4), "LabelTask queue undoing failed."
    label_task.undo()
    next_task = next(label_task)
    assert next_task == (0, 1), "LabelTask queue undoing failed."
