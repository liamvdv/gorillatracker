from pathlib import Path

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video


def video_tracking_frame_features_query(video: Path) -> Select[tuple[TrackingFrameFeature]]:
    return (
        select(TrackingFrameFeature)
        .select_from(Video)
        .where(Video.filename == str(video.name))
        .join(Tracking)
        .join(TrackingFrameFeature)
    )


def load_video_by_filename(session: Session, video: Path) -> Video:
    return session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()