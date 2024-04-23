# HACK(memben): DO NOT PUSH THIS CODE TO MAIN


from functools import cache

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature

DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"


def hacked_get_clique_graph() -> IndexedCliqueGraph[TrackingFrameFeature]:
    engine = create_engine(DB_URI)
    session_cls = sessionmaker(bind=engine)
    with session_cls() as session:
        tracked_features = list(
            session.execute(
                select(TrackingFrameFeature)
                .where(
                    TrackingFrameFeature.cache_path.isnot(None),
                    TrackingFrameFeature.tracking_id.isnot(None),
                    TrackingFrameFeature.feature_type == "body",
                )
                .order_by(TrackingFrameFeature.tracking_id)
            )
            .scalars()
            .all()
        )
        graph: IndexedCliqueGraph[TrackingFrameFeature] = IndexedCliqueGraph(tracked_features)  # type: ignore
        # merge within the same tracking_id
        for i in range(1, len(tracked_features)):
            if tracked_features[i].tracking_id == tracked_features[i - 1].tracking_id:
                graph.merge(tracked_features[i], tracked_features[i - 1])

        # partition by tracking_id
        reps = {f.tracking_id: f for f in tracked_features}
        for reps_id in reps:
            for other_id in reps:
                if reps_id == other_id:
                    continue
                graph.partition(reps[reps_id], reps[other_id])

    return graph


if __name__ == "__main__":
    hacked_get_clique_graph()
