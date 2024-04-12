import pytest

from gorillatracker.ssl_pipeline.data_structures import EdgeType, UnionFind, UnionGraph


@pytest.fixture
def setup_union_find() -> UnionFind[int]:
    return UnionFind(list(range(10)))


@pytest.fixture
def setup_union_graph() -> UnionGraph[int]:
    union_graph = UnionGraph(list(range(5)))
    union_graph.add_edge(0, 1, EdgeType.POSITIVE)
    union_graph.add_edge(1, 2, EdgeType.POSITIVE)
    union_graph.add_edge(2, 3, EdgeType.NEGATIVE)
    union_graph.add_edge(3, 4, EdgeType.NEGATIVE)
    return union_graph


def test_union_find_union_and_find(setup_union_find: UnionFind[int]) -> None:
    uf = setup_union_find
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2), "UnionFind union and find operations failed."
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3), "UnionFind union and find operations failed."


def test_union_graph_group_relationship(setup_union_graph: UnionGraph[int]) -> None:
    u_graph = setup_union_graph
    assert u_graph.has_positive_relationship(0, 1), "UnionGraph positive relationship check failed."
    assert not u_graph.has_negative_relationship(0, 1), "UnionGraph positive relationship check failed."
    assert not u_graph.has_positive_relationship(0, 3), "UnionGraph negative relationship check failed."
    assert u_graph.has_negative_relationship(0, 3), "UnionGraph negative relationship check failed."

    assert not u_graph.has_positive_relationship(0, 4), "UnionGraph negative relationship check failed."

    assert u_graph.get_entities_in_group(0) == set([0, 1, 2]), "UnionGraph group relationship check failed."
    assert u_graph.get_entities_in_group(1) == set([0, 1, 2]), "UnionGraph group relationship check failed."
    assert u_graph.get_entities_in_group(2) == set([0, 1, 2]), "UnionGraph group relationship check failed."
    assert u_graph.get_entities_in_group(3) == set([3]), "UnionGraph group relationship check failed."
    assert u_graph.get_entities_in_group(4) == set([4]), "UnionGraph group relationship check failed."


def test_union_merge_groups(setup_union_graph: UnionGraph[int]) -> None:
    u_graph = setup_union_graph
    assert not u_graph.has_positive_relationship(0, 4), "UnionGraph negative relationship check failed."
    u_graph.add_edge(1, 4, EdgeType.POSITIVE)
    assert u_graph.has_positive_relationship(0, 4), "UnionGraph merge groups failed."


def test_union_graph_fails_invalid_edge(setup_union_graph: UnionGraph[int]) -> None:
    u_graph = setup_union_graph
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 0, EdgeType.NEGATIVE)
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 1, EdgeType.NEGATIVE)
    with pytest.raises(AssertionError):
        u_graph.add_edge(0, 3, EdgeType.POSITIVE)
