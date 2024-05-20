import datetime as dt
from typing import Optional

from sqlalchemy import Select, func, select

from gorillatracker.ssl_pipeline.models import Video

def video_count_filter(
    query: Select[tuple[Video]], num_videos: int
) -> Select[tuple[Video]]:
    """Filter the query to return a specific number of videos."""
    return query.limit(num_videos)

def camera_with_videos_filter(query: Select[tuple[Video]], num_videos: int) -> Select[tuple[Video]]:
    """Filter the query to return videos from cameras with at least a certain number of videos."""
    video_count = func.count(Video.video_id)
    video_count_subquery = (
        select(
            Video.camera_id,
            video_count
        )
        .group_by(Video.camera_id)
        .having(video_count >= num_videos)
    ).subquery()
    return query.join(video_count_subquery, Video.camera_id == video_count_subquery.c.camera_id)
    
def extract_year(dt_obj: Optional[dt.datetime]) -> int:
    if dt_obj is None:
        return -1
    return dt_obj.year

def year_filter(
    query: Select[tuple[Video]], year: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific year."""
    return query.where(extract_year(Video.start_time) == year)

def extract_hour(dt_obj: Optional[dt.datetime])-> int:
    if dt_obj is None:
        return -1
    return dt_obj.hour

def daytime_filter(
    query: Select[tuple[Video]], start_hour: int, end_hour: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific time of the day."""
    return query.where(start_hour <= extract_hour(Video.start_time) <= end_hour)

def extract_month(dt_obj: Optional[dt.datetime]) -> int:
    if dt_obj is None:
        return -1
    return dt_obj.month

def month_filter(
    query: Select[tuple[Video]], start_month: int, end_month: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific months."""
    return query.where(start_month <= extract_month(Video.start_time) <= end_month)