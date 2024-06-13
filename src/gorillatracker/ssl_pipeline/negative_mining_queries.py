import datetime as dt
from typing import Callable, Sequence

from sqlalchemy import ColumnElement, Select, alias, func, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import Camera, Tracking, TrackingFrameFeature, Video, VideoFeature


def build_overlapping_trackings_query(video_ids: Sequence[int]) -> Select[tuple[int, int]]:
    tracking_frame_feature_cte = (
        select(TrackingFrameFeature.tracking_id, TrackingFrameFeature.frame_nr, TrackingFrameFeature.video_id)
        .where(TrackingFrameFeature.video_id.in_(video_ids))
        .cte("tracking_frame_feature")
    )

    tracking_summary_cte = (
        select(
            tracking_frame_feature_cte.c.tracking_id,
            func.min(tracking_frame_feature_cte.c.frame_nr).label("min_frame_nr"),
            func.max(tracking_frame_feature_cte.c.frame_nr).label("max_frame_nr"),
            tracking_frame_feature_cte.c.video_id,
        )
        .where(tracking_frame_feature_cte.c.tracking_id.isnot(None))
        .group_by(TrackingFrameFeature.tracking_id, TrackingFrameFeature.video_id)
        .cte("tracking_summary")
    )

    left_summary = aliased(tracking_summary_cte, name="anon_1")
    right_summary = aliased(tracking_summary_cte, name="anon_2")

    # Main query to find overlapping trackings
    stmt = (
        select(left_summary.c.tracking_id, right_summary.c.tracking_id)
        .join(right_summary, left_summary.c.video_id == right_summary.c.video_id)
        .where(
            (left_summary.c.min_frame_nr <= right_summary.c.max_frame_nr)
            & (right_summary.c.min_frame_nr <= left_summary.c.max_frame_nr)
            & (left_summary.c.tracking_id < right_summary.c.tracking_id)
        )
    )
    return stmt


def find_overlapping_trackings(session: Session, video_ids: Sequence[int]) -> Sequence[tuple[int, int]]:
    negative_tuples = fetch_negative_tuples(session, video_ids, build_overlapping_trackings_query)
    return negative_tuples


def tracking_ids_from_videos(video_ids: Sequence[int]) -> Select[tuple[int]]:
    return (
        select(Tracking.tracking_id)
        .where(Video.video_id.in_(video_ids))
        .join(Video, Tracking.video_id == Video.video_id)
    )


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


def social_group_negatives(video_ids: Sequence[int]) -> Select[tuple[int, int]]:
    relevant_videos_cte = select(Video.video_id).where(Video.video_id.in_(video_ids)).cte("relevant_videos")

    video_social_groups_cte = (
        select(relevant_videos_cte.c.video_id, VideoFeature.value)
        .join(VideoFeature, relevant_videos_cte.c.video_id == VideoFeature.video_id)
        .where(VideoFeature.feature_type == "social_group")
        .cte("video_social_groups")
    )

    left_cte = aliased(video_social_groups_cte, name="left_cte")
    right_cte = aliased(video_social_groups_cte, name="right_cte")

    stmt = (
        select(left_cte.c.video_id, right_cte.c.video_id)
        .join(right_cte, left_cte.c.video_id < right_cte.c.video_id)
        .where(left_cte.c.value != right_cte.c.value)
    )
    return stmt


def find_social_group_negatives(session: Session, video_ids: Sequence[int]) -> Sequence[tuple[int, int]]:
    negative_tuples = fetch_negative_tuples(session, video_ids, social_group_negatives)
    return negative_tuples


def fetch_negative_tuples(
    session: Session, video_ids: Sequence[int], query_builder: Callable[[Sequence[int]], Select[tuple[int, int]]]
) -> Sequence[tuple[int, int]]:
    BATCH_SIZE = 200
    num_batches = len(video_ids) // BATCH_SIZE
    negative_tuples = []
    for i in range(num_batches + 1):
        batch_video_ids = video_ids[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        stmt = query_builder(batch_video_ids)
        result = session.execute(stmt).all()
        negative_tuples.extend([(row[0], row[1]) for row in result])
    return negative_tuples


if __name__ == "__main__":
    import time

    from sqlalchemy import create_engine

    from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ

    engine = create_engine(GorillaDatasetKISZ.DB_URI)
    video_ids = list(range(1, 10000))
    with Session(engine) as session:
        start = time.time()
        video_negatives = find_social_group_negatives(session, video_ids)
        end = time.time()
        print(f"Found {len(video_negatives)} video negatives in {end - start:.2f} seconds")
