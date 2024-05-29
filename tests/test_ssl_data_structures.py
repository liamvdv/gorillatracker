import pytest

from gorillatracker.ssl_pipeline.data_structures import (
    CliqueGraph,
    IndexedCliqueGraph,
    MultiLayerCliqueGraph,
    UnionFind,
)


@pytest.fixture
def setup_union_find() -> UnionFind[int]:
    return UnionFind(list(range(10)))


@pytest.fixture
def setup_clique_graph() -> CliqueGraph[int]:
    clique_graph = CliqueGraph(list(range(5)))
    clique_graph.merge(0, 1)
    clique_graph.merge(1, 2)
    clique_graph.partition(2, 3)
    clique_graph.partition(3, 4)
    return clique_graph


@pytest.fixture
def setup_indexed_clique_graph() -> IndexedCliqueGraph[int]:
    vertices = [3, 1, 0, 4, 2]  # Intentionally unsorted to test sorting within class
    indexed_clique_graph = IndexedCliqueGraph(vertices)
    indexed_clique_graph.merge(0, 1)
    indexed_clique_graph.partition(1, 2)
    indexed_clique_graph.partition(2, 3)
    return indexed_clique_graph


def test_union_find_union_and_find(setup_union_find: UnionFind[int]) -> None:
    uf = setup_union_find
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2), "UnionFind union and find operations failed."
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3), "UnionFind union and find operations failed."


def test_clique_graph_group_relationship(setup_clique_graph: CliqueGraph[int]) -> None:
    c_graph = setup_clique_graph
    assert c_graph.is_connected(0, 1), "CliqueGraph positive relationship check failed."
    assert not c_graph.is_partitioned(0, 1), "CliqueGraph positive relationship check failed."
    assert not c_graph.is_connected(0, 3), "CliqueGraph negative relationship check failed."
    assert c_graph.is_partitioned(0, 3), "CliqueGraph negative relationship check failed."

    assert not c_graph.is_connected(0, 4), "CliqueGraph negative relationship check failed."

    assert set(c_graph.get_clique(0)) == {
        0,
        1,
        2,
    }, "CliqueGraph group relationship check failed."
    assert set(c_graph.get_clique(1)) == {
        0,
        1,
        2,
    }, "CliqueGraph group relationship check failed."
    assert set(c_graph.get_clique(2)) == {
        0,
        1,
        2,
    }, "CliqueGraph group relationship check failed."
    assert set(c_graph.get_clique(3)) == {3}, "CliqueGraph group relationship check failed."
    assert set(c_graph.get_clique(4)) == {4}, "CliqueGraph group relationship check failed."


def test_clique_graph_merge_groups(setup_clique_graph: CliqueGraph[int]) -> None:
    c_graph = setup_clique_graph
    assert not c_graph.is_connected(0, 4), "CliqueGraph negative relationship check failed."
    c_graph.merge(1, 4)
    assert c_graph.is_connected(0, 4), "CliqueGraph merge groups failed."


def test_clique_graph_fails_invalid_edge(setup_clique_graph: CliqueGraph[int]) -> None:
    c_graph = setup_clique_graph
    with pytest.raises(AssertionError):
        c_graph.partition(0, 0)
    with pytest.raises(AssertionError):
        c_graph.partition(0, 1)
    with pytest.raises(AssertionError):
        c_graph.merge(0, 3)


def test_indexed_clique_graph_clique_representative(
    setup_indexed_clique_graph: IndexedCliqueGraph[int],
) -> None:
    icg = setup_indexed_clique_graph
    assert icg.get_clique_representative(1) == 0, "Incorrect clique representative."
    assert icg.get_clique_representative(3) == 3, "Incorrect clique representative."
    icg.merge(0, 3)
    assert icg.get_clique_representative(3) == 0, "Incorrect clique representative."


def test_indexed_clique_graph_adjacent_cliques(
    setup_indexed_clique_graph: IndexedCliqueGraph[int],
) -> None:
    icg = setup_indexed_clique_graph
    icg.partition(1, 3)
    adjacent_cliques = icg.get_adjacent_cliques(0)
    assert 3 in adjacent_cliques, "Adjacent cliques not identified correctly."
    assert set(adjacent_cliques[3]) == {3}, "Adjacent clique members incorrect."


@pytest.fixture
def two_layer_clique_graph() -> MultiLayerCliqueGraph[int]:
    """Prefixed numbers always describe the relationship to their parent clique. For example, 11, 12 is a child of 1."""
    first_layer = CliqueGraph([1, 2, 3])
    first_layer.partition(1, 2)
    second_layer = MultiLayerCliqueGraph(
        [11, 12, 13, 21, 22, 23, 31, 32, 33],
        first_layer,
        {11: 1, 12: 1, 13: 1, 21: 2, 22: 2, 23: 2, 31: 3, 32: 3, 33: 3},
    )
    second_layer.partition(11, 12)
    second_layer.partition(21, 22)
    second_layer.merge(12, 13)
    return second_layer


@pytest.fixture
def three_layer_clique_graph(
    two_layer_clique_graph: MultiLayerCliqueGraph[int],
) -> MultiLayerCliqueGraph[int]:
    second_layer = two_layer_clique_graph
    third_layer = MultiLayerCliqueGraph(
        [111, 112, 113, 121, 122, 211, 212, 213, 311, 312, 313],
        second_layer,
        {111: 11, 121: 12, 211: 21, 311: 31, 313: 31},
    )
    third_layer.merge(111, 112)
    third_layer.merge(112, 113)
    third_layer.merge(121, 122)
    third_layer.merge(211, 212)
    third_layer.merge(212, 213)
    third_layer.merge(311, 312)
    third_layer.partition(312, 313)

    return third_layer


def test_two_layer_clique_graph_connections(two_layer_clique_graph: MultiLayerCliqueGraph[int]) -> None:
    tlcg = two_layer_clique_graph
    assert tlcg.is_partitioned(11, 12)
    assert not tlcg.is_connected(11, 21)
    assert tlcg.is_connected(12, 13)
    assert not tlcg.is_partitioned(12, 13)
    assert tlcg.is_partitioned(11, 21)
    assert not tlcg.is_connected(11, 22)


def test_three_layer_clique_graph_connections(three_layer_clique_graph: MultiLayerCliqueGraph[int]) -> None:
    ttcg = three_layer_clique_graph
    assert ttcg.is_connected(111, 112)
    assert ttcg.is_connected(111, 113)
    assert ttcg.is_connected(211, 212)
    assert ttcg.is_connected(311, 312)

    assert ttcg.is_partitioned(311, 313)

    # Check that partitioning works from the middle layer
    assert ttcg.is_partitioned(111, 121)

    # Check that partitioning works from the top down
    assert ttcg.is_partitioned(111, 211)
    assert ttcg.is_partitioned(111, 212)
