"""
This module contains pre-defined database queries.
"""

import datetime as dt
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

from sqlalchemy import Select, alias, and_, func, or_, select, update
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Camera, Task, TaskStatus, TaskType, TrackingFrameFeature, Video

log = logging.getLogger(__name__)

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


def multiple_videos_filter(video_ids: list[int]) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances from the specified videos.

    Equivalent to python:
    ```python
    def filter(self, video_ids: list[int]) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.tracking.video_id in video_ids, frame_features)
    ```
    """
    # NOTE(memben): Alternative of using a intermediate Video result and joining it, turned out to be not a difference
    return select(TrackingFrameFeature).where(TrackingFrameFeature.video_id.in_(video_ids))


def associated_filter(query: Select[tuple[TrackingFrameFeature]]) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances that are associated with a tracking.
    """
    return query.where(TrackingFrameFeature.tracking_id.isnot(None))


def cached_filter(query: Select[tuple[TrackingFrameFeature]]) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances that were cropped.
    """
    return query.where(TrackingFrameFeature.cached)


def min_count_filter(
    query: Select[tuple[TrackingFrameFeature]], min_feature_count: int
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances that belong to trackings with at least `min_feature_count` features of the specified `feature_type`.
    If `feature_type` is None, considers all feature types.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        tracking_id_grouped = group_by_tracking_id(list(frame_features))
        predicate = (
            lambda features: len([x for x in features if x.feature_type == self.feature_type]) >= self.min_feature_count
            if self.feature_type is not None
            else len(features) >= self.min_feature_count
        )
        return chain.from_iterable(
            features for features in tracking_id_grouped.values() if predicate(features)
        )

    ```
    """

    alias_query = alias(query.subquery())
    subquery = (
        select(
            alias_query.c.tracking_id,
            func.count().label("feature_count"),
        ).group_by(alias_query.c.tracking_id)
    ).subquery()

    query = query.join(subquery, subquery.c.tracking_id == TrackingFrameFeature.tracking_id)
    query = query.where(subquery.c.feature_count >= min_feature_count)
    return query


def feature_type_filter(
    query: Select[tuple[TrackingFrameFeature]], feature_types: list[str]
) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances with the specified `feature_types`.

    Equivalent to python:
    ```python
    def filter(self, frame_features: Iterator[TrackingFrameFeature]) -> Iterator[TrackingFrameFeature]:
        return filter(lambda x: x.feature_type in self.feature_types, frame_features)
    ```
    """
    if "body_with_face" in feature_types:
        return bodies_with_face_filter(query)
    return query.where(TrackingFrameFeature.feature_type.in_(feature_types))


def bodies_with_face_filter(query: Select[tuple[TrackingFrameFeature]]) -> Select[tuple[TrackingFrameFeature]]:
    """
    Filters the query to include only TrackingFrameFeature instances with a body feature and a face feature in the same frame.
    """
    subquery = (
        select(
            TrackingFrameFeature.tracking_id,
            TrackingFrameFeature.frame_nr,
        ).where(TrackingFrameFeature.feature_type == "face_90")
    ).subquery()
    query = query.where(
        TrackingFrameFeature.feature_type == "body",
        TrackingFrameFeature.tracking_id == subquery.c.tracking_id,
        TrackingFrameFeature.frame_nr == subquery.c.frame_nr,
    )
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


def bbox_filter(
    query: Select[tuple[TrackingFrameFeature]],
    min_width: Optional[int],
    max_width: Optional[int],
    min_height: Optional[int],
    max_height: Optional[int],
) -> Select[tuple[TrackingFrameFeature]]:
    if min_width is not None:
        query = query.where(TrackingFrameFeature.bbox_width >= min_width)
    if max_width is not None:
        query = query.where(TrackingFrameFeature.bbox_width <= max_width)
    if min_height is not None:
        query = query.where(TrackingFrameFeature.bbox_height >= min_height)
    if max_height is not None:
        query = query.where(TrackingFrameFeature.bbox_height <= max_height)
    return query


def load_features(session: Session, video_id: int, feature_types: list[str]) -> Sequence[TrackingFrameFeature]:
    stmt = feature_type_filter(video_filter(video_id), feature_types)
    return session.execute(stmt).scalars().all()


def load_tracked_features(session: Session, video_id: int, feature_types: list[str]) -> Sequence[TrackingFrameFeature]:
    stmt = feature_type_filter(associated_filter(video_filter(video_id)), feature_types)
    return session.execute(stmt).scalars().all()


def load_video(session: Session, video_path: Path, version: str) -> Video:
    return session.execute(
        select(Video).where(Video.absolute_path == str(video_path), Video.version == version)
    ).scalar_one()


def load_videos(session: Session, video_paths: list[Path], version: str) -> Sequence[Video]:
    return (
        session.execute(
            select(Video).where(
                Video.absolute_path.in_([str(video_path) for video_path in video_paths]), Video.version == version
            )
        )
        .scalars()
        .all()
    )


def load_preprocessed_videos(session: Session, version: str) -> Sequence[Video]:
    stmt = select(Video).where(Video.version == version)
    return session.execute(stmt).scalars().all()


def get_or_create_camera(session: Session, camera_name: str) -> Camera:
    camera = session.execute(select(Camera).where(Camera.name == camera_name)).scalar_one_or_none()
    if camera is None:
        camera = Camera(name=camera_name)
        session.add(camera)
        session.commit()
    return camera


def get_next_task(
    session: Session,
    task_type: TaskType,
    max_retries: int = 0,
    task_timeout: dt.timedelta = dt.timedelta(days=1),
    task_subtype: str = "",
) -> Iterator[Task]:
    """Yields the next task. Useable in a multiprocessing context.
    Use transactional_task to properly handle the task in a transactional manner.

    Args:
        session (Session): The database session.
        task_type (str): The type of the task.
        max_retries (int): The maximum number of retries, for failed or timed out tasks. Defaults to 0.
        task_timeout (dt.timedelta): The maximum time a task can be in processing state before being considered timed out. Defaults to one day.
    """
    while True:
        timeout_threshold = dt.datetime.now(dt.timezone.utc) - task_timeout
        pending_condition = Task.status == TaskStatus.PENDING
        processing_condition = (
            (Task.status == TaskStatus.PROCESSING)
            & (Task.updated_at < timeout_threshold)
            & (Task.retries < max_retries)
        )
        failed_condition = (Task.status == TaskStatus.FAILED) & (Task.retries < max_retries)

        stmt = (
            select(Task)
            .where(
                Task.task_type == task_type,
                Task.task_subtype == task_subtype,
                or_(pending_condition, processing_condition, failed_condition),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        task = session.execute(stmt).scalars().first()
        if task is None:
            break

        if task.status != TaskStatus.PENDING:
            task.retries += 1
        task.status = TaskStatus.PROCESSING
        session.commit()
        yield task


@contextmanager
def transactional_task(session: Session, task: Task) -> Iterator[Task]:
    """Each session is committed after a successful task completion,
    and rolled back if an exception is raised by this function.
    Do **not** commit any changes that should be rolled back on exception."""
    try:
        yield task
    except Exception as e:
        session.rollback()
        task.status = TaskStatus.FAILED
        session.commit()
        log.exception(e)
        # NOTE(memben): swallow
    else:
        task.status = TaskStatus.COMPLETED
        session.commit()


def reset_dependent_tasks_status(session: Session, dependent: TaskType, provider: TaskType) -> None:
    """Resets all dependent task status where the provider had one or more retries"""
    subquery = select(Task.video_id).where(Task.retries > 0, Task.task_type == provider)

    stmt = (
        update(Task)
        .where(and_(Task.task_type == dependent, Task.video_id.in_(subquery)))
        .values(status=TaskStatus.PENDING)
    )

    session.execute(stmt)
    session.commit()


if __name__ == "__main__":
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(os.environ.get("POSTGRESQL_URI") or "sqlite:///:memory:")

    session_cls = sessionmaker(bind=engine)

    with Session(engine) as session:
        video_ids = list(session.execute(select(Video.video_id)).scalars().all())
        query = multiple_videos_filter(video_ids[:200])
        query = associated_filter(query)
        query = confidence_filter(query, 0.5)
        query = bbox_filter(query, 10, 100, 10, 100)
        query = min_count_filter(query, 10)
        tffs = session.execute(query).scalars().all()
        # print out some bbox widths and heights
        for tff in tffs:
            print(tff.bbox_width, tff.bbox_height)
