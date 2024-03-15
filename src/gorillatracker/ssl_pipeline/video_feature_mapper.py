import logging
from itertools import zip_longest
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.correlators import Correlator
from gorillatracker.ssl_pipeline.helpers import AssociatedBoundingBox, BoundingBox, get_tracked_frames
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


def predict_correlate_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    type: str,
    session_cls: sessionmaker[Session],
    correlate_features: Correlator,
    DANGER_activate_visual_debugging: DEBUG = None,
) -> None:
    with session_cls() as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()

        assert video_tracking.frame_step == yolo_kwargs.get(
            "vid_stride", 1
        ), "vid_stride must match the frame_step of the body tracking"

        id_to_tracking = {tracking.tracking_id: tracking for tracking in video_tracking.trackings}

        unresolved_boxes: list[BoundingBox] = []
        tracked_frames = get_tracked_frames(session, video_tracking, filter_by_type="body")
        prediction: results.Results
        # TODO(memben) change back to zip_longest
        for prediction, tracked_frame in zip(yolo_model.predict(video, stream=True, **yolo_kwargs), tracked_frames):
            assert prediction is not None
            assert tracked_frame is not None

            detections = prediction.boxes
            assert isinstance(detections, results.Boxes)
            boxes: list[BoundingBox] = []
            for detection in detections:
                x, y, w, h = detection.xywhn[0].tolist()
                c = detection.conf.item()
                boxes.append(BoundingBox(x, y, w, h, c, video_tracking.width, video_tracking.height))
            tracked_boxes: list[AssociatedBoundingBox] = [
                AssociatedBoundingBox(feature.tracking_id, BoundingBox.from_tracking_frame_feature(feature))
                for feature in tracked_frame.frame_features
            ]
            correlated_boxes, uncorrelated_boxes = correlate_features(tracked_boxes, boxes, threshold=0.1)
            unresolved_boxes.extend(uncorrelated_boxes)

            for correlated_box in correlated_boxes:
                tracking_id = correlated_box.association
                frame_nr = tracked_frame.frame_nr
                bbox = correlated_box.bbox
                TrackingFrameFeature(
                    tracking=id_to_tracking[
                        tracking_id
                    ],  # NOTE needed for data model validation and adding it implicitly to the DB
                    frame_nr=frame_nr,
                    bbox_x_center=bbox.x_center_n,
                    bbox_y_center=bbox.y_center_n,
                    bbox_width=bbox.width_n,
                    bbox_height=bbox.height_n,
                    confidence=bbox.confidence,
                    type=type,
                )

        log.info(f"Unresolved boxes: {len(unresolved_boxes)} with the type: {type} for video {video.name}")

        if DANGER_activate_visual_debugging:
            log.warning("DANGER: Introducing undefined state, accepting danger")
            dummy_tracking = Tracking(
                video=video_tracking,
                tracking_id=(-1) * video_tracking.video_id,
            )
            for unresolved_box in unresolved_boxes:
                frame_nr = tracked_frame.frame_nr
                bbox = unresolved_box
                session.add(
                    TrackingFrameFeature(
                        tracking=dummy_tracking,
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
