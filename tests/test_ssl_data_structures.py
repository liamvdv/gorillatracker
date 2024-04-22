import pytest

from gorillatracker.ssl_pipeline.data_structures import CliqueGraph, UnionFind


@pytest.fixture
def setup_union_find() -> UnionFind[int]:
    return UnionFind(list(range(10)))


@pytest.fixture
def setup_union_graph() -> CliqueGraph[int]:
    union_graph = CliqueGraph(list(range(5)))
    union_graph.merge(0, 1)
    union_graph.merge(1, 2)
    union_graph.partition(2, 3)
    union_graph.partition(3, 4)
    return union_graph


def test_union_find_union_and_find(setup_union_find: UnionFind[int]) -> None:
    uf = setup_union_find
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2), "UnionFind union and find operations failed."
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3), "UnionFind union and find operations failed."


def test_union_graph_group_relationship(setup_union_graph: CliqueGraph[int]) -> None:
    u_graph = setup_union_graph
    assert u_graph.is_connected(0, 1), "CliqueGraph positive relationship check failed."
    assert not u_graph.is_partitioned(0, 1), "CliqueGraph positive relationship check failed."
    assert not u_graph.is_connected(0, 3), "CliqueGraph negative relationship check failed."
    assert u_graph.is_partitioned(0, 3), "CliqueGraph negative relationship check failed."

    assert not u_graph.is_connected(0, 4), "CliqueGraph negative relationship check failed."

    assert u_graph.get_clique(0) == {0, 1, 2}, "CliqueGraph group relationship check failed."
    assert u_graph.get_clique(1) == {0, 1, 2}, "CliqueGraph group relationship check failed."
    assert u_graph.get_clique(2) == {0, 1, 2}, "CliqueGraph group relationship check failed."
    assert u_graph.get_clique(3) == {3}, "CliqueGraph group relationship check failed."
    assert u_graph.get_clique(4) == {4}, "CliqueGraph group relationship check failed."


def test_union_merge_groups(setup_union_graph: CliqueGraph[int]) -> None:
    u_graph = setup_union_graph
    assert not u_graph.is_connected(0, 4), "CliqueGraph negative relationship check failed."
    u_graph.merge(1, 4)
    assert u_graph.is_connected(0, 4), "CliqueGraph merge groups failed."


def test_union_graph_fails_invalid_edge(setup_union_graph: CliqueGraph[int]) -> None:
    u_graph = setup_union_graph
    with pytest.raises(AssertionError):
        u_graph.partition(0, 0)
    with pytest.raises(AssertionError):
        u_graph.partition(0, 1)
    with pytest.raises(AssertionError):
        u_graph.merge(0, 3)
