import logging
from itertools import zip_longest
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.correlators import Correlator
from gorillatracker.ssl_pipeline.helpers import (
    AssociatedBoundingBox,
    BoundingBox,
    TrackedFrame,
    load_tracked_frames,
    load_video_tracking,
)
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)

DEBUG = Literal["INTRODUCING UNDEFINED STATE, ACCEPTING DANGER", None]
"""
Important: This debug setting is strictly for debugging and should not be used in production environments. 
It temporarily allows the addition of dummy trackings with negative tracking_ids (negated video_id) to visualize unresolved boxes, 
thus breaching referential integrity. This is meant only for early pipeline stages (tracking, correlating, visualization). 
Also, it necessitates disabling the TrackingFrameFeature table's unique constraint on ("tracking_id", "frame_nr", "type").

To revert any changes made using this in production, delete all Tracking entries with negative tracking_id.
"""


def convert_to_associated_bbox(tracked_frame: TrackedFrame) -> list[AssociatedBoundingBox]:
    return [
        AssociatedBoundingBox(feature.tracking_id, BoundingBox.from_tracking_frame_feature(feature))
        for feature in tracked_frame.frame_features
    ]


def process_detections(prediction: results.Results, video_tracking: Video) -> list[BoundingBox]:
    assert isinstance(prediction.boxes, results.Boxes)
    return [
        BoundingBox(x, y, w, h, c, video_tracking.width, video_tracking.height)
        for x, y, w, h, c in (detection.xywhn[0].tolist() + [detection.conf.item()] for detection in prediction.boxes)
    ]


def add_unresolved_boxes_for_debugging(
    session: Session,
    framewise_unresolved_boxes: list[tuple[int, list[BoundingBox]]],
    dummy_tracking: Tracking,
    type: str,
) -> None:
    session.add_all(
        TrackingFrameFeature(
            tracking=dummy_tracking,
            frame_nr=frame_nr,
            bbox_x_center=box.x_center_n,
            bbox_y_center=box.y_center_n,
            bbox_width=box.width_n,
            bbox_height=box.height_n,
            confidence=box.confidence,
            type=type,
        )
        for frame_nr, boxes in framewise_unresolved_boxes
        for box in boxes
    )


def store_correlated_boxes(
    session: Session,
    correlated_boxes: list[AssociatedBoundingBox],
    frame_nr: int,
    type: str,
    id_to_tracking: dict[int, Tracking],
) -> None:
    session.add_all(
        TrackingFrameFeature(
            tracking=id_to_tracking[box.association],
            frame_nr=frame_nr,
            bbox_x_center=box.bbox.x_center_n,
            bbox_y_center=box.bbox.y_center_n,
            bbox_width=box.bbox.width_n,
            bbox_height=box.bbox.height_n,
            confidence=box.bbox.confidence,
            type=type,
        )
        for box in correlated_boxes
    )


def predict_correlate_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    type: str,
    session_cls: sessionmaker[Session],
    correlate: Correlator,
    DANGER_activate_visual_debugging: DEBUG = None,
) -> None:
    with session_cls() as session:
        video_tracking = load_video_tracking(session, video)
        assert video_tracking.frame_step == yolo_kwargs.get(
            "vid_stride", 1
        ), "vid_stride must match the frame_step of the body tracking"

        id_to_tracking = {tracking.tracking_id: tracking for tracking in video_tracking.trackings}
        framewise_unresolved_boxes: list[tuple[int, list[BoundingBox]]] = []

        tracked_frames = load_tracked_frames(session, video_tracking, filter_by_type="body")[
            1:
        ]  # NOTE(memben): YOLOv8 skips the 0th frame https://github.com/ultralytics/ultralytics/issues/8976

        prediction: results.Results
        for prediction, tracked_frame in zip_longest(
            yolo_model.predict(video, stream=True, **yolo_kwargs), tracked_frames
        ):
            assert prediction is not None
            assert tracked_frame is not None
            feature_bboxes = process_detections(prediction, video_tracking)
            body_bboxes = convert_to_associated_bbox(tracked_frame)
            correlated_boxes, uncorrelated_boxes = correlate(body_bboxes, feature_bboxes, threshold=0.1)
            framewise_unresolved_boxes.append((tracked_frame.frame_nr, uncorrelated_boxes))
            store_correlated_boxes(session, correlated_boxes, tracked_frame.frame_nr, type, id_to_tracking)

        total_unresolved_boxes = sum(len(boxes) for _, boxes in framewise_unresolved_boxes)
        log.info(f"{total_unresolved_boxes} unresolved boxes of type {type} in video {video.name} ")
        session.commit()

        if DANGER_activate_visual_debugging:
            log.warning("DANGER: Visual Debugging flag set, introducing undefined state, accepting danger")
            dummy_tracking = Tracking(
                video=video_tracking,  # NOTE needed for data model validation
                tracking_id=(-1) * video_tracking.video_id,
            )
            session.add(dummy_tracking)
            add_unresolved_boxes_for_debugging(session, framewise_unresolved_boxes, dummy_tracking, type)
            session.commit()
