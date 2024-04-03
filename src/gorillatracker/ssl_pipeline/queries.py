"""
This module contains pre-defined database queries.
"""

from pathlib import Path

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

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


def video_filter(video: Path) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances from the specified video.

    Equivalent to python:
    ```python
    def filter(self, video: Path) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.tracking.video.filename == video.name, frame_features)
    ```
    """
    return select(TrackingFrameFeature).join(Tracking).join(Video).where(Video.filename == str(video.name))


def min_count_filter(
    query: Select[tuple[TrackingFrameFeature]], min_feature_count: int, feature_type: str | None = None
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
    query = query.where(TrackingFrameFeature.type.in_(feature_types))
    return query


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


def load_video_by_filename(session: Session, video: Path) -> Video:
    return session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
