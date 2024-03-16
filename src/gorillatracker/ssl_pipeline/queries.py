from pathlib import Path
from typing import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video


# TODO (memben): figure out how to type this
def video_tracking_frame_features_query(video_id: int) -> Select:  # type: ignore
    return (
        select(TrackingFrameFeature)
        .select_from(Video)
        .where(Video.video_id == video_id)
        .join(Tracking)
        .join(TrackingFrameFeature)
    )


def load_video_by_filename(session: Session, video_path: Path) -> Video:
    return session.execute(select(Video).where(Video.filename == str(video_path.name))).scalar_one()


def load_video_tracking_frame_features(session: Session, video_id: int) -> Sequence[TrackingFrameFeature]:
    return session.execute(video_tracking_frame_features_query(video_id)).scalars().all()
