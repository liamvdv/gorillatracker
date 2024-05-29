"""
This module contains pre-defined database queries.
"""

import datetime as dt
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

from sqlalchemy import ColumnElement, Select, alias, and_, func, or_, select, update
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import (
    Camera,
    Task,
    TaskStatus,
    TaskType,
    Tracking,
    TrackingFrameFeature,
    Video,
    VideoFeature,
)

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
            lambda features: len([x for x in features if x.feature_type == self.feature_type]) >= self.min_feature_count
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
        subquery = subquery.where(TrackingFrameFeature.feature_type == feature_type)

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
        return filter(lambda x: x.feature_type in self.feature_types, frame_features)
    ```
    """
    return query.where(TrackingFrameFeature.feature_type.in_(feature_types))


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


def time_diff(
    left_datetime: ColumnElement[dt.datetime], right_datetime: ColumnElement[dt.datetime]
) -> ColumnElement[float]:
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
        .where(Video.version == version, VideoFeature.feature_type == "social_group")
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
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres")

    session_cls = sessionmaker(bind=engine)

    with session_cls() as session:
        reset_dependent_tasks_status(session, TaskType.CORRELATE, TaskType.TRACK)
