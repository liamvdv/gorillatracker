from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.create_ssl_dataset import filter_and_extract
from gorillatracker.ssl_pipeline.models import Base
from gorillatracker.ssl_pipeline.track_videos import track_and_store


def ssl_pipeline(video_path: str, session: Session, base_dir: str, n_images: int) -> None:
    """
    The main pipeline for the semi-supervised learning pipeline.

    Args:
        video_path (str): The path to the video to track.
        base_dir (str): The base directory to save the frames to. Will be saved to base_dir/camera_id/video_id/tracked_animal_id
        n_images (int): The number of images to extract.
    """
    video_id = track_and_store(video_path, session)
    filter_and_extract(video_id, session, base_dir, n_images)


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.close()
    Base.metadata.drop_all(engine)
