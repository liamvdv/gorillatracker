"""
This module contains pre-defined database queries.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy import ColumnElement, Select, alias, func, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import (
    Camera,
    ProcessedVideoFrameFeature,
    Tracking,
    TrackingFrameFeature,
    Video,
    VideoFeature,
)

"""
The helper function `group_by_tracking_id` is not used perse, but it is included here for completeness.

```python
def group_by_tracking_id(frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
    frame_features.sort(key=lambda x: x.tracking.tracking_id)
    return {
        tracking_id: list(features)
        for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
    }
```

"""


def video_filter(video_id: int) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances from the specified video.

    Equivalent to python:
    ```python
    def filter(self, video_id: int) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.tracking.video_id == video_id, frame_features)
    ```
    """
    return select(TrackingFrameFeature).where(TrackingFrameFeature.video_id == video_id)


def associated_filter(query: Select[tuple[TrackingFrameFeature]]) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances that are associated with a tracking.
    """
    return query.where(TrackingFrameFeature.tracking_id.isnot(None))


def min_count_filter(
    query: Select[tuple[TrackingFrameFeature]], min_feature_count: int, feature_type: Optional[str] = None
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances that belong to trackings with at least `min_feature_count` features of the specified `feature_type`.
    If `feature_type` is None, considers all feature types.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        tracking_id_grouped = group_by_tracking_id(list(frame_features))
        predicate = (
            lambda features: len([x for x in features if x.type == self.feature_type]) >= self.min_feature_count
            if self.feature_type is not None
            else len(features) >= self.min_feature_count
        )
        return chain.from_iterable(
            features for features in tracking_id_grouped.values() if predicate(features)
        )

    ```
    """
    subquery = (
        select(TrackingFrameFeature.tracking_id)
        .group_by(TrackingFrameFeature.tracking_id)
        .having(func.count(TrackingFrameFeature.tracking_id) >= min_feature_count)
    )

    if feature_type is not None:
        subquery = subquery.where(TrackingFrameFeature.type == feature_type)

    query = query.where(TrackingFrameFeature.tracking_id.in_(subquery))

    return query


def feature_type_filter(
    query: Select[tuple[TrackingFrameFeature]], feature_types: list[str]
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances with the specified `feature_types`.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.type in self.feature_types, frame_features)
    ```
    """
    return query.where(TrackingFrameFeature.type.in_(feature_types))


def confidence_filter(
    query: Select[tuple[TrackingFrameFeature]], min_confidence: float
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances with a confidence greater than or equal to `min_confidence`.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.confidence >= self.min_confidence, frame_features)
    ```
    """
    query = query.where(TrackingFrameFeature.confidence >= min_confidence)
    return query


def load_features(session: Session, video_id: int, feature_types: list[str]) -> Sequence[TrackingFrameFeature]:
    stmt = feature_type_filter(video_filter(video_id), feature_types)
    return session.execute(stmt).scalars().all()


def load_tracked_features(session: Session, video_id: int, feature_types: list[str]) -> Sequence[TrackingFrameFeature]:
    stmt = feature_type_filter(associated_filter(video_filter(video_id)), feature_types)
    return session.execute(stmt).scalars().all()


def load_video(session: Session, video_path: Path, version: str) -> Video:
    return session.execute(select(Video).where(Video.path == str(video_path), Video.version == version)).scalar_one()


def load_videos(session: Session, video_paths: list[Path], version: str) -> Sequence[Video]:
    return (
        session.execute(
            select(Video).where(
                Video.path.in_([str(video_path) for video_path in video_paths]), Video.version == version
            )
        )
        .scalars()
        .all()
    )


def load_processed_videos(session: Session, version: str, required_feature_types: list[str]) -> Sequence[Video]:
    stmt = select(Video).where(Video.version == version)
    if required_feature_types:
        stmt = (
            stmt.join(ProcessedVideoFrameFeature)
            .where(ProcessedVideoFrameFeature.type.in_(required_feature_types))
            .group_by(Video.video_id)
            .having(func.count(ProcessedVideoFrameFeature.type.distinct()) == len(required_feature_types))
        )
    return session.execute(stmt).scalars().all()


def get_or_create_camera(session: Session, camera_name: str) -> Camera:
    camera = session.execute(select(Camera).where(Camera.name == camera_name)).scalar_one_or_none()
    if camera is None:
        camera = Camera(name=camera_name)
        session.add(camera)
        session.commit()
    return camera


def find_overlapping_trackings(session: Session) -> Sequence[tuple[Tracking, Tracking]]:
    subquery = (
        select(
            TrackingFrameFeature.tracking_id,
            func.min(TrackingFrameFeature.frame_nr).label("min_frame_nr"),
            func.max(TrackingFrameFeature.frame_nr).label("max_frame_nr"),
            TrackingFrameFeature.video_id,
        )
        .where(TrackingFrameFeature.tracking_id.isnot(None))
        .group_by(TrackingFrameFeature.tracking_id)
    ).subquery()

    left_subquery = alias(subquery)
    right_subquery = alias(subquery)

    left_tracking = aliased(Tracking)
    right_tracking = aliased(Tracking)

    stmt = (
        select(left_tracking, right_tracking)
        .join(left_subquery, left_tracking.tracking_id == left_subquery.c.tracking_id)
        .join(right_subquery, right_tracking.tracking_id == right_subquery.c.tracking_id)
        .where(
            (left_subquery.c.min_frame_nr <= right_subquery.c.max_frame_nr)
            & (right_subquery.c.min_frame_nr <= left_subquery.c.max_frame_nr)
            & (left_subquery.c.video_id == right_subquery.c.video_id)
            & (left_subquery.c.tracking_id < right_subquery.c.tracking_id)
        )
    )

    overlapping_trackings = session.execute(stmt).fetchall()
    return [(row[0], row[1]) for row in overlapping_trackings]


def great_circle_distance(
    left_latitude: ColumnElement[float],
    left_longitude: ColumnElement[float],
    right_latitude: ColumnElement[float],
    right_longitude: ColumnElement[float],
) -> ColumnElement[float]:
    return 6371 * func.acos(
        func.cos(func.radians(left_latitude))
        * func.cos(func.radians(right_latitude))
        * func.cos(func.radians(right_longitude) - func.radians(left_longitude))
        + func.sin(func.radians(left_latitude)) * func.sin(func.radians(right_latitude))
    )


def time_diff(left_datetime: ColumnElement[datetime], right_datetime: ColumnElement[datetime]) -> ColumnElement[float]:
    return func.abs(func.julianday(left_datetime) - func.julianday(right_datetime)) * 24


def travel_time(
    left_latitude: ColumnElement[float],
    left_longitude: ColumnElement[float],
    right_latitude: ColumnElement[float],
    right_longitude: ColumnElement[float],
    travel_speed: float,
) -> ColumnElement[float]:
    return great_circle_distance(left_latitude, left_longitude, right_latitude, right_longitude) / travel_speed


def travel_distance_negatives(session: Session, version: str, travel_speed: float) -> Sequence[tuple[Video, Video]]:
    # join video table with camera table and select video_id, camera_id, latitude, and longitude
    subquery = (
        select(Video.video_id, Video.camera_id, Camera.latitude, Camera.longitude, Video.start_time)
        .join(Camera, Video.camera_id == Camera.camera_id)
        .where(Video.version == version)
    ).subquery()

    left_subquery = alias(subquery)
    right_subquery = alias(subquery)

    left_video = aliased(Video)
    right_video = aliased(Video)

    stmt = (
        select(left_video, right_video)
        .join(left_subquery, left_video.video_id == left_subquery.c.video_id)
        .join(right_subquery, right_video.video_id == right_subquery.c.video_id)
        .where(
            left_subquery.c.camera_id != right_subquery.c.camera_id,
            travel_time(
                left_subquery.c.latitude,
                left_subquery.c.longitude,
                right_subquery.c.latitude,
                right_subquery.c.longitude,
                travel_speed,
            )
            > time_diff(left_subquery.c.start_time, right_subquery.c.start_time),
            left_subquery.c.video_id < right_subquery.c.video_id,
        )
    )

    result = session.execute(stmt).all()
    negative_tuples = [(row[0], row[1]) for row in result]
    return negative_tuples


def social_group_negatives(session: Session, version: str) -> Sequence[tuple[Video, Video]]:
    subquery = (
        select(Video.video_id, VideoFeature.value)
        .join(VideoFeature, Video.video_id == VideoFeature.video_id)
        .where(Video.version == version, VideoFeature.type == "Social Group")
        # Note: string can change
    ).subquery()

    left_subquery = alias(subquery)
    right_subquery = alias(subquery)

    left_video = aliased(Video)
    right_video = aliased(Video)

    stmt = (
        select(left_video, right_video)
        .join(left_subquery, left_video.video_id == left_subquery.c.video_id)
        .join(right_subquery, right_video.video_id == right_subquery.c.video_id)
        .where(left_subquery.c.value != right_subquery.c.value, left_subquery.c.video_id < right_subquery.c.video_id)
    )
    result = session.execute(stmt).all()
    negative_tuples = [(row[0], row[1]) for row in result]
    return negative_tuples


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///test.db")

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-09"

    with session_cls() as session:
        video_negatives = social_group_negatives(session, version)
        print(video_negatives[:10])
        social_groups = session.execute(select(VideoFeature).where(VideoFeature.type == "Social Group")).all()
        print(social_groups)
        video_negatives = travel_distance_negatives(session, version, 10)
        print(video_negatives[:10])
