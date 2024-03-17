from pathlib import Path

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video


def video_tracking_frame_features_query(video: Path) -> Select[tuple[TrackingFrameFeature]]:
    return (
        select(TrackingFrameFeature)
        .select_from(Video)
        .where(Video.filename == str(video.name))
        .join(Tracking)
        .join(TrackingFrameFeature)
    )


def video_tracking_frame_features_min_count_query(
    video: Path, min_feature_count: int, feature_types: list[str] | None = None
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Returns a query to select TrackingFrameFeature instances that have at least `min_feature_count` features.
    Considers only the features of the specified `feature_types` for counting if provided.
    """

    tracking_frame_feature = aliased(TrackingFrameFeature)

    subquery = (
        select(tracking_frame_feature.tracking_id)
        .join(Tracking)
        .join(Video)
        .where(Video.filename == str(video.name))
        .group_by(tracking_frame_feature.tracking_id)
        .having(func.count(tracking_frame_feature.tracking_id) >= min_feature_count)
    )

    if feature_types:
        subquery = subquery.where(tracking_frame_feature.type.in_(feature_types))

    subquery = subquery.subquery()

    query = select(TrackingFrameFeature).join(subquery, TrackingFrameFeature.tracking_id == subquery.c.tracking_id)

    return query


def load_video_by_filename(session: Session, video: Path) -> Video:
    return session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()


print(str(video_tracking_frame_features_min_count_query(Path("test"), 10)))
