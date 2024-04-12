from sqlalchemy.orm import Session
from typing import Sequence

from gorillatracker.ssl_pipeline.models import Tracking
from gorillatracker.ssl_pipeline.queries import find_overlapping_trackings


def create_union_graph(session: Session, trackings: Sequence[tuple[Tracking, Tracking]]):
    pass


def negative_miner(session: Session):
    trackings = find_overlapping_trackings(session)
    
    graph = create_union_graph(session, trackings)
    


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///test.db")
    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    negative_miner(session)
    