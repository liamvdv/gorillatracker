from sqlalchemy import Integer, Select, and_, case, extract, func, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Video,Camera

def get_camera_ids(session: Session) -> list[int]:
    camera_ids = session.execute(select(Camera.camera_id)).scalars().all()
    return camera_ids

def get_video_query(version: str) -> Select[tuple[Video]]:
    query = select(Video).where(Video.version == version)
    return query

def get_videos_from_query(query: Select[tuple[Video]], session: Session) -> list[Video]:
    result = session.execute(query).scalars().all()
    video_list = [video for video in result]
    return video_list

def video_count_filter(query: Select[tuple[Video]], num_videos: int) -> Select[tuple[Video]]:
    """Filter the query to return a specific number of videos."""
    return query.limit(num_videos)

def camera_with_videos_filter(query: Select[tuple[Video]], num_videos: int) -> Select[tuple[Video]]:
    """Filter the query to return videos from cameras with at least a certain number of videos."""
    video_count = func.count(Video.video_id)
    video_count_subquery = (
        select(Video.camera_id, video_count).group_by(Video.camera_id).having(video_count >= num_videos)
    ).subquery()
    return query.join(video_count_subquery, Video.camera_id == video_count_subquery.c.camera_id)

def year_filter(query: Select[tuple[Video]], years: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific year."""
    return query.where(extract("year", Video.start_time) in years)


def hour_filter(query: Select[tuple[Video]], hours: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific time of the day."""
    return query.where(extract("hour", Video.start_time) in hours)


def month_filter(query: Select[tuple[Video]], months: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific months."""
    return query.where(extract("month", Video.start_time) in months)


def video_length_filter(query: Select[tuple[Video]], min_length: int, max_length: int) -> Select[tuple[Video]]:
    """Filter the query to return videos within a specific length range."""
    duration_seconds = func.cast(case((Video.fps != 0, Video.frames / Video.fps), else_=0), Integer)
    return query.where(and_(min_length <= duration_seconds, duration_seconds <= max_length))

def camera_id_filter(query: Select[tuple[Video]], camera_ids: list[int]) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific cameras."""
    return query.where(Video.camera_id in (camera_ids))


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("db-uri-here")
    version = "2024-04-18"

    session_cls = sessionmaker(bind=engine)
    
    with session_cls() as session:
        query = get_video_query(version)
        # add filter queries here
        videos = get_videos_from_query(query, session)
        print(len(videos))
        for video in videos[:20]:
            print(video)
