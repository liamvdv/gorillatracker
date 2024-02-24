from sqlalchemy.orm import Session


def filter_and_extract(video_id: int, session: Session, base_dir: str, n_images: int) -> None:
    """
    Filter and extract frames from a video.

    Args:
        video_id (int): The id of the video to extract frames from.
        session (Session): The database session.
        base_dir (str): The base directory to save the frames to. Will be saved to base_dir/camera_id/video_id/tracked_animal_id
        n_images (int): The number of images to extract.
    """
    pass
