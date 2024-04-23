# NOTE(memben): let's worry about how we parse configs from the yaml file later

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph
from gorillatracker.ssl_pipeline.dataset import LazyImage
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature

DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"


def DEMO_get_clique_graph() -> IndexedCliqueGraph[LazyImage]:
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

        lazy_images = [LazyImage(f.tracking_frame_feature_id, f.cache_path) for f in tracked_features]  # type: ignore
        graph = IndexedCliqueGraph(lazy_images)

        # merge within the same tracking_id
        for i in range(1, len(tracked_features)):
            if tracked_features[i].tracking_id == tracked_features[i - 1].tracking_id:
                graph.merge(lazy_images[i], lazy_images[i - 1])

        # partition by tracking_id
        tracking_id_mapper = {f.tracking_id: i for i, f in enumerate(tracked_features)}
        for tracking_id in tracking_id_mapper:
            for other_tracking_id in tracking_id_mapper:
                if tracking_id != other_tracking_id:
                    graph.partition(
                        lazy_images[tracking_id_mapper[tracking_id]], lazy_images[tracking_id_mapper[other_tracking_id]]
                    )

        return graph


if __name__ == "__main__":
    DEMO_get_clique_graph()
