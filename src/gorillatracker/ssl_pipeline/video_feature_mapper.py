from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.helpers import BoundingBox, get_tracked_frames
from gorillatracker.ssl_pipeline.models import Video


@dataclass(frozen=True)
class BoundingBoxWithID:
    id: int
    bbox: BoundingBox


def predict_correlate_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    session_cls: sessionmaker[Session],
    feature_correlator: Callable[[list[BoundingBoxWithID], list[BoundingBox]], list[BoundingBoxWithID]],
) -> None:
    with session_cls() as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        assert video_tracking.frame_step == yolo_kwargs.get(
            "vid_stride", 1
        ), "vid_stride must match the frame_step of the body tracking"

        tracked_frames = get_tracked_frames(session, video_tracking, filter_by_type="body")
        result: results.Results
        for result, tracked_frame in zip(
            yolo_model.predict(video, stream=True, **yolo_kwargs), tracked_frames, strict=True
        ):
            pass
