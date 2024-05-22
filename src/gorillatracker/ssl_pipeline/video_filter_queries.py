from sqlalchemy import Select, func, select, extract, and_, case, Integer

from gorillatracker.ssl_pipeline.models import Video


def get_videos(session, version: str):
    query = select(Video).where(Video.version == version)
    # query = video_count_filter(query, 150000)
    # query = year_filter(query, 2019)
    # query = daytime_filter(query, 0, 24)
    # query = month_filter(query, 7, 7)
    query = video_length_filter(query, 1, 1)
    query = camera_with_videos_filter(query, 5000)
    result = session.execute(query).scalars().all()
    video_list = [video.duration for video in result]
    return video_list

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

def year_filter(
    query: Select[tuple[Video]], year: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific year."""
    return query.where(extract("year", Video.start_time) == year)

def daytime_filter(
    query: Select[tuple[Video]], start_hour: int, end_hour: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from a specific time of the day."""
    return query.where(and_(start_hour <= extract("hour", Video.start_time), extract("hour", Video.start_time) <= end_hour))

def month_filter(
    query: Select[tuple[Video]], start_month: int, end_month: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos from specific months."""
    return query.where(and_(start_month <= extract("month", Video.start_time), extract("month", Video.start_time) <= end_month))

def video_length_filter(
    query: Select[tuple[Video]], min_length: int, max_length: int
) -> Select[tuple[Video]]:
    """Filter the query to return videos within a specific length range."""
    duration_seconds = func.cast(
        case(
            (Video.fps != 0, Video.frames / Video.fps),
            else_=0
        ), Integer
    )
    return query.where(
        and_(min_length <= duration_seconds, duration_seconds <= max_length)
    )

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql+psycopg2://postgres:HyfCW95WnwmXmnQpBmiw@10.149.20.40:5432/postgres")

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-18"

    with session_cls() as session:
        videos = get_videos(session, version)
        print(len(videos))
        