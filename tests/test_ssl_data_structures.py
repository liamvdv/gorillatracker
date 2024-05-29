import pytest

from gorillatracker.ssl_pipeline.data_structures import CliqueGraph, IndexedCliqueGraph, UnionFind


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
