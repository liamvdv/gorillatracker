from sqlalchemy.orm import Session
from typing import Sequence

from gorillatracker.ssl_pipeline.models import Tracking
from gorillatracker.ssl_pipeline.queries import find_overlapping_trackings
from gorillatracker.ssl_pipeline.data_structures import UnionGraph, EdgeType


def create_union_graph(session: Session, trackings: Sequence[tuple[Tracking, Tracking]]) -> UnionGraph[Tracking]:
    union_graph = UnionGraph(session.query(Tracking).all())
    for left_tracking, right_tracking in trackings:
        union_graph.add_edge(left_tracking, right_tracking, EdgeType.NEGATIVE)
    return union_graph


def negative_miner(session: Session):
    trackings = find_overlapping_trackings(session)
    union_graph = create_union_graph(session, trackings)
    return union_graph

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///test.db")
    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    graph = negative_miner(session)
    print(graph.negative_relations)
    