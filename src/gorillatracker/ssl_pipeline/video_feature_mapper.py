from pathlib import Path
from typing import Any, Callable, Generator

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.helpers import BoundingBox, TrackedBoundingBox, get_tracked_frames
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video


def predict_correlate_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    type: str,
    session_cls: sessionmaker[Session],
    feature_correlator: Callable[[list[TrackedBoundingBox], list[BoundingBox]], list[TrackedBoundingBox]],
) -> None:
    with session_cls() as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        assert video_tracking.frame_step == yolo_kwargs.get(
            "vid_stride", 1
        ), "vid_stride must match the frame_step of the body tracking"

        tracked_frames = get_tracked_frames(session, video_tracking, filter_by_type="body")
        predictions: Generator[results.Results, None, None] = yolo_model.predict(video, stream=True, **yolo_kwargs)
        for prediction, tracked_frame in zip(predictions, tracked_frames, strict=True):
            detections = prediction.boxes
            assert isinstance(detections, results.Boxes)
            boxes: list[BoundingBox] = []
            for detection in detections:
                x, y, w, h = detection.xywhn[0].tolist()
                c = detection.conf.item()
                boxes.append(BoundingBox(x, y, w, h, c, video_tracking.width, video_tracking.height))
            tracked_boxes: list[TrackedBoundingBox] = [
                TrackedBoundingBox(feature.tracking_id, BoundingBox.from_tracking_frame_feature(feature))
                for feature in tracked_frame.frame_features
            ]
            correlated_boxes = feature_correlator(tracked_boxes, boxes)
            for correlated_box in correlated_boxes:
                tracking_id = correlated_box.id
                frame_nr = tracked_frame.frame_nr
                bbox = correlated_box.bbox
                session.add(
                    TrackingFrameFeature(
                        tracking_id=tracking_id,
                        frame_nr=frame_nr,
                        bbox_x_center=bbox.x_center_n,
                        bbox_y_center=bbox.y_center_n,
                        bbox_width=bbox.width_n,
                        bbox_height=bbox.height_n,
                        confidence=bbox.confidence,
                        type=type,
                    )
                )
        session.commit()
