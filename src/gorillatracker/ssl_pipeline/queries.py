"""
This module contains pre-defined database queries.
"""

import datetime as dt
from pathlib import Path
from typing import Iterator, Optional, Sequence

from sqlalchemy import Select, alias, func, or_, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import Camera, Task, TaskStatus, TaskType, Tracking, TrackingFrameFeature, Video

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


def load_processed_videos(session: Session, version: str, required_completed_tasks: list[str]) -> Sequence[Video]:
    stmt = select(Video).where(Video.version == version)
    if required_completed_tasks:
        stmt = (
            stmt.join(Task)
            .where(Task.task_type.in_(required_completed_tasks), Task.status == TaskStatus.COMPLETED)
            .group_by(Video.video_id)
            .having(func.count(Task.task_type.distinct()) == len(required_completed_tasks))
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


def get_next_task(
    session: Session,
    task_type: TaskType,
    max_retries: int = 0,
    task_timeout: dt.timedelta = dt.timedelta(days=1),
    task_subtype: str = "",
) -> Iterator[Task]:
    """Yields and handles task in a transactional manner. Useable in a multiprocessing context.
    Each session is committed after a successful task completion, and rolled back if an exception is raised by this function.
    Do **not** commit any changes that should be rolled back on exception.

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
            .with_for_update(skip_locked=True)
        )

        task = session.execute(stmt).scalars().first()
        if task is None:
            break

        if task.status != TaskStatus.PENDING:
            task.retries += 1
        task.status = TaskStatus.PROCESSING
        session.commit()

        try:
            yield task
        except Exception:
            session.rollback()
            task.status = TaskStatus.FAILED
            session.commit()
            raise
        else:
            task.status = TaskStatus.COMPLETED
            session.commit()


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///test.db")

    session_cls = sessionmaker(bind=engine)

    # find first video_id in the database and then find overlapping trackings for that video and print them
    with session_cls() as session:
        overlapping_trackings = find_overlapping_trackings(session)
        for left_tracking, right_tracking in overlapping_trackings:
            print(left_tracking.tracking_id, right_tracking.tracking_id)
