from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO

from gorillatracker.ssl_pipeline.models import Video


def predict_correlate_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    session_cls: sessionmaker[Session],
    feature_correlator: Callable,
) -> None:
    with session_cls() as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        assert video_tracking.frame_step == yolo_kwargs.get(
            "vid_stride", 1
        ), "vid_stride must match the frame_step in the database"
