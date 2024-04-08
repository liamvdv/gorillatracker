"""
This module contains pre-defined database queries.
"""

from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import ProcessedVideoFrameFeature, TrackingFrameFeature, Video

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
