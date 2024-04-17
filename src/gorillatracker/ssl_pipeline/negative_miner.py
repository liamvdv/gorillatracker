from typing import Sequence

from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.data_structures import CliqueGraph
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import find_overlapping_trackings


def create_tracking_clique_graph(session: Session, trackings: Sequence[tuple[Tracking, Tracking]]) -> CliqueGraph[Tracking]:
    clique_graph = CliqueGraph(session.query(Tracking).all()) # select TODO
    for left_tracking, right_tracking in trackings:
        clique_graph.partition(left_tracking, right_tracking)
    return clique_graph


def create_tracking_frame_clique_graph(session: Session, tracking_graph: CliqueGraph[Tracking]) -> CliqueGraph[TrackingFrameFeature]:
    tracking_frame_graph = CliqueGraph(session.query(TrackingFrameFeature).all()) # select TODO
    # first we put all trackingframefeatures with the same tracking_id in the same clique
    
    # second we partition the cliques based on CliqueGraph of Trackings
    
    return tracking_frame_graph


def negative_miner(session: Session):
    trackings = find_overlapping_trackings(session)
    clique_graph = create_tracking_clique_graph(session, trackings)
    return clique_graph



if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///test.db")
    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    graph = negative_miner(session)
    # select first tracking from the database
    trackings = session.query(Tracking)
    tracking = trackings[2]
    assert tracking is not None
    print(graph.get_clique(tracking))
    print(graph.get_adjacent_cliques(tracking))
