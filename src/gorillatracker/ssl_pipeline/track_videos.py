from sqlalchemy.orm import Session


def track_and_store(video_path: str, session: Session) -> int:
    """
    Track and store the tracked animals in the database.

    Args:
        video_path (str): The path to the video to track.
        session (Session): The database session.

    Returns:
        int: The id of the video in the database.
    """
    raise NotImplementedError
