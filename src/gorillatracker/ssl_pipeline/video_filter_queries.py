from typing import Union

from sqlalchemy import Integer, Select, and_, case, extract, func, or_, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Camera, Video


def get_camera_ids(session: Session) -> list[int]:
    camera_ids = session.execute(select(Camera.camera_id)).scalars().all()
    return list(camera_ids)


def get_video_query(version: str) -> Select[tuple[Video]]:
    query = select(Video).where(Video.version == version)
    return query


def get_videos_from_query(query: Select[tuple[Video]], session: Session) -> list[int]:
    result = session.execute(query).scalars().all()
    video_list = [video.video_id for video in result]
    return video_list


def video_count_filter(query: Select[tuple[Video]], num_videos: int) -> Select[tuple[Video]]:
    """Filter the query to return a specific number of videos."""
    return query.limit(num_videos)


def year_filter(query: Select[tuple[Video]], years: list[Union[int, None]]) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific year."""
    if None in years:
        years = [year for year in years if year is not None]
        return query.where(or_(extract("year", Video.start_time).in_(years), Video.start_time.is_(None)))
    else:
        return query.where(extract("year", Video.start_time).in_(years))


def month_filter(query: Select[tuple[Video]], months: list[Union[int, None]]) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific months."""
    if None in months:
        months = [month for month in months if month is not None]
        return query.where(or_(extract("month", Video.start_time).in_(months), Video.start_time.is_(None)))
    else:
        return query.where(extract("month", Video.start_time).in_(months))


def hour_filter(query: Select[tuple[Video]], hours: list[Union[int, None]]) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific time of the day."""
    if None in hours:
        hours = [hour for hour in hours if hour is not None]
        return query.where(or_(extract("hour", Video.start_time).in_(hours), Video.start_time.is_(None)))
    else:
        return query.where(extract("hour", Video.start_time).in_(hours))


def video_length_filter(query: Select[tuple[Video]], min_length: int, max_length: int) -> Select[tuple[Video]]:
    """Filter the query to return videos within a specific length range."""
    duration_seconds = func.cast(case((Video.fps != 0, Video.frames / Video.fps), else_=0), Integer)
    return query.where(and_(min_length <= duration_seconds, duration_seconds <= max_length))


def camera_id_filter(query: Select[tuple[Video]], camera_ids: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific cameras."""
    return query.where(Video.camera_id.in_(camera_ids))


def video_not_in(query: Select[tuple[Video]], video_ids: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos not in the list of video_ids."""
    return query.where(Video.video_id.notin_(video_ids))


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("db_uri")
    version = "2024-04-18"

    session_cls = sessionmaker(bind=engine)

    with session_cls() as session:
        query = get_video_query(version)
        # add filter queries here
        videos = get_videos_from_query(query, session)
        print(len(videos))
        for video in videos[:20]:
            print(video)
