import datetime as dt
from typing import Sequence

from sqlalchemy import ColumnElement, alias, func, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import Camera, TrackingFrameFeature, Video, VideoFeature


def find_overlapping_trackings(session: Session) -> Sequence[tuple[int, int]]:
    subquery = (
        select(
            TrackingFrameFeature.tracking_id,
            func.min(TrackingFrameFeature.frame_nr).label("min_frame_nr"),
            func.max(TrackingFrameFeature.frame_nr).label("max_frame_nr"),
            TrackingFrameFeature.video_id,
        )
        .where(TrackingFrameFeature.tracking_id.isnot(None))
        .group_by(TrackingFrameFeature.tracking_id, TrackingFrameFeature.video_id)
    ).subquery()

    left_subquery = alias(subquery)
    right_subquery = alias(subquery)

    stmt = (
        select(left_subquery.c.tracking_id, right_subquery.c.tracking_id)
        .join(right_subquery, left_subquery.c.video_id == right_subquery.c.video_id)
        .where(
            (left_subquery.c.min_frame_nr <= right_subquery.c.max_frame_nr)
            & (right_subquery.c.min_frame_nr <= left_subquery.c.max_frame_nr)
            & (left_subquery.c.tracking_id < right_subquery.c.tracking_id)
        )
    )
    print("Starting query...")
    overlapping_trackings = session.execute(stmt).fetchall()
    print("Done")
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
