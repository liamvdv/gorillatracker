"""
This module contains pre-defined database queries.
"""

from pathlib import Path

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import lateral, true

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

"""
The helper function `group_by_tracking_id` is not used in the codebase, but it is included here for completeness.

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


def random_sampling_filter(
    query: Select[tuple[TrackingFrameFeature]], n_images_per_tracking: int, seed: int | None = None
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Modifies the query to randomly sample a subset of TrackingFrameFeature instances per tracking.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        tracking_id_grouped = group_by_tracking_id(list(frame_features))
        for features in tracking_id_grouped.values():
            num_samples = min(len(features), self.n_images_per_tracking)
            yield from random.sample(features, num_samples)
    ```
    """
    TrackingFrameFeatureAlias = aliased(TrackingFrameFeature)

    subquery = (
        select(TrackingFrameFeatureAlias)
        .where(TrackingFrameFeatureAlias.tracking_id == Tracking.tracking_id)
        .order_by(func.random())
        .limit(10)
        .subquery()
    )


    query = select(TrackingFrameFeatureAlias).join(lateral(subquery), true())

    return query


def load_video_by_filename(session: Session, video: Path) -> Video:
    return session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
