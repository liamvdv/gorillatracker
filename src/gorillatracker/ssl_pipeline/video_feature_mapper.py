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
    load_tracked_frames,
    load_video_tracking,
)
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)

DEBUG = Literal["INTRODUCING UNDEFINED STATE, ACCEPTING DANGER", None]
"""
Read carefully:
You should never ever use this in production.
Only use this for debugging purposes.
---
The system is designed to have referential integrity.
However, to visualize the unresolved boxes, we need to add them to the database.
Thus we introduce a dummy tracking with a negative tracking_id (which is the video_id negated).
This is a dangerous operation, as it introduces an undefined state in the database.
It should work for the first part of the pipeline (tracking and correlating and visualization).
To allow multiple TrackingFrameFeatures we need to deactivate the unique constraint on the TrackingFrameFeature table.
`(UniqueConstraint("tracking_id", "frame_nr", "type")`.

Recovery: Delete all Tracking with a negative tracking_id.
"""


def get_id_to_tracking(trackings: list[Tracking]) -> dict[int, Tracking]:
    return {tracking.tracking_id: tracking for tracking in trackings}


def process_detections(prediction: results.Results, video_tracking: Video) -> list[BoundingBox]:
    assert isinstance(prediction.boxes, results.Boxes)
    return [
        BoundingBox(x, y, w, h, c, video_tracking.width, video_tracking.height)
        for x, y, w, h, c in (detection.xywhn[0].tolist() + [detection.conf.item()] for detection in prediction.boxes)
    ]


def add_unresolved_boxes_for_debugging(
    session: Session, unresolved_frame_boxes: list[tuple[int, list[BoundingBox]]], dummy_tracking: Tracking, type: str
) -> None:
    for frame_nr, unresolved_boxes in unresolved_frame_boxes:
        for unresolved_box in unresolved_boxes:
            session.add(
                TrackingFrameFeature(
                    tracking=dummy_tracking,
                    frame_nr=frame_nr,
                    bbox_x_center=unresolved_box.x_center_n,
                    bbox_y_center=unresolved_box.y_center_n,
                    bbox_width=unresolved_box.width_n,
                    bbox_height=unresolved_box.height_n,
                    confidence=unresolved_box.confidence,
                    type=type,
                )
            )


def store_correlated_boxes(
    session: Session,
    correlated_boxes: list[AssociatedBoundingBox],
    frame_nr: int,
    type: str,
    id_to_tracking: dict[int, Tracking],
) -> None:
    for correlated_box in correlated_boxes:
        bbox = correlated_box.bbox
        session.add(
            TrackingFrameFeature(
                tracking=id_to_tracking[correlated_box.association],  # NOTE needed for data model validation
                frame_nr=frame_nr,
                bbox_x_center=bbox.x_center_n,
                bbox_y_center=bbox.y_center_n,
                bbox_width=bbox.width_n,
                bbox_height=bbox.height_n,
                confidence=bbox.confidence,
                type=type,
            )
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

        id_to_tracking = get_id_to_tracking(video_tracking.trackings)
        unresolved_frame_boxes: list[tuple[int, list[BoundingBox]]] = []
        tracked_frames = load_tracked_frames(session, video_tracking, filter_by_type="body")
        tracked_frames = tracked_frames[1:]  # NOTE(memben): https://github.com/ultralytics/ultralytics/issues/8976
        prediction: results.Results
        for prediction, tracked_frame in zip_longest(
            yolo_model.predict(video, stream=True, **yolo_kwargs), tracked_frames
        ):
            assert prediction is not None
            assert tracked_frame is not None

            detected_bboxes = process_detections(prediction, video_tracking)
            tracked_boxes = [
                AssociatedBoundingBox(feature.tracking_id, BoundingBox.from_tracking_frame_feature(feature))
                for feature in tracked_frame.frame_features
            ]
            correlated_boxes, uncorrelated_boxes = correlate(tracked_boxes, detected_bboxes, threshold=0.1)
            unresolved_frame_boxes.append((tracked_frame.frame_nr, uncorrelated_boxes))

            store_correlated_boxes(session, correlated_boxes, tracked_frame.frame_nr, type, id_to_tracking)

        log.info(
            f"Unresolved boxes: {sum(len(boxes) for _, boxes in unresolved_frame_boxes)} in {sum(1 for _, boxes in unresolved_frame_boxes if boxes)} frames in video {video.name}, type {type}"
        )
        session.commit()

        if DANGER_activate_visual_debugging:
            log.warning("DANGER: Visual Debugging flag set, introducing undefined state, accepting danger")
            dummy_tracking = Tracking(
                video=video_tracking,  # NOTE needed for data model validation
                tracking_id=(-1) * video_tracking.video_id,
            )
            session.add(dummy_tracking)
            add_unresolved_boxes_for_debugging(session, unresolved_frame_boxes, dummy_tracking, type)
            session.commit()
