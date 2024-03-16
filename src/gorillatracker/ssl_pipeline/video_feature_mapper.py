import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.correlators import Correlator
from gorillatracker.ssl_pipeline.helpers import (
    AssociatedBoundingBox,
    BoundingBox,
    TrackedFrame,
    load_tracked_frames,
    load_video_tracking,
    video_reader,
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


def process_prediction(prediction: results.Results, video_tracking: Video) -> list[BoundingBox]:
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
        tracked_frames = load_tracked_frames(session, video_tracking, filter_by_type="body")

        # NOTE(memben): YOLOv8s video streaming has an error https://github.com/ultralytics/ultralytics/issues/8976, so we fix it internally
        with video_reader(video, frame_step=video_tracking.frame_step) as video_feed:
            for video_frame, tracked_frame in zip_longest(video_feed, tracked_frames):
                assert video_frame is not None
                assert tracked_frame is not None
                assert video_frame.frame_nr == tracked_frame.frame_nr
                predictions: list[results.Results] = yolo_model.predict(video_frame.frame, **yolo_kwargs)
                assert len(predictions) == 1
                feature_bboxes = process_prediction(predictions[0], video_tracking)
                body_bboxes = convert_to_associated_bbox(tracked_frame)
                correlated_boxes, uncorrelated_boxes = correlate(body_bboxes, feature_bboxes, threshold=0.1)
                framewise_unresolved_boxes.append((tracked_frame.frame_nr, uncorrelated_boxes))
                store_correlated_boxes(session, correlated_boxes, tracked_frame.frame_nr, type, id_to_tracking)

            total_unresolved_boxes = sum(len(boxes) for _, boxes in framewise_unresolved_boxes)
            log.info(f"{total_unresolved_boxes} unresolved boxes of type {type} in video {video.name} ")
        session.commit()

        if DANGER_activate_visual_debugging:
            log.warning("DANGER: Visual Debugging flag set, introducing undefined state, accepting danger")

            dummy_tracking_id = (-1) * video_tracking.video_id
            dummy_tracking = session.get(Tracking, dummy_tracking_id)

            if dummy_tracking is None:
                dummy_tracking = Tracking(video=video_tracking, tracking_id=dummy_tracking_id)
                session.add(dummy_tracking)

            add_unresolved_boxes_for_debugging(session, framewise_unresolved_boxes, dummy_tracking, type)
            session.commit()


_yolo_model = None
_yolo_kwargs = None
_type = None
_session_cls = None
_correlate = None


def _init_predictor(
    yolo_model: Path,
    yolo_kwargs: dict[str, Any],
    type: str,
    engine: Engine,
    correlate: Correlator,
    # TODO(memben): figure out why typing makes it fail
    gpu_queue: Queue,  # type: ignore
) -> None:
    log = logging.getLogger(__name__)
    global _yolo_model, _yolo_kwargs, _type, _session_cls, _correlate
    _yolo_model = YOLO(yolo_model)
    _yolo_kwargs = yolo_kwargs
    _type = type
    _correlate = correlate

    assigned_gpu = gpu_queue.get()
    log.info(f"Predictor initialized on GPU {assigned_gpu}")
    if "device" in yolo_kwargs:
        raise ValueError("device will be overwritten by the assigned GPU")
    yolo_kwargs["device"] = f"cuda:{assigned_gpu}"

    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    _session_cls = sessionmaker(bind=engine)


def _multiprocess_predict_correlate_store(video: Path) -> None:
    global _yolo_model, _yolo_kwargs, _type, _session_cls, _correlate
    assert _yolo_model is not None, "Predictor is not initialized, call init_predictor first"
    assert _yolo_kwargs is not None, "YOLO kwargs are not initialized, use init_predictor instead"
    assert _type is not None, "Type is not initialized, use init_predictor instead"
    assert _session_cls is not None, "Session class is not initialized, use init_predictor instead"
    assert _correlate is not None, "Correlator is not initialized, use init_predictor instead"
    predict_correlate_store(
        video,
        _yolo_model,
        _yolo_kwargs,
        _type,
        _session_cls,
        _correlate,
    )


def multiproces_feature_mapping(
    yolo_model: Path,
    yolo_kwargs: dict[str, Any],
    type: str,
    videos: list[Path],
    engine: Engine,
    correlate: Correlator,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    with Session(engine) as session:
        processed_videos = session.execute(select(Video.filename)).scalars().all()
        assert all(video.name in list(processed_videos) for video in videos), "Not all videos have been processed yet"

    gpu_queue: Queue[int] = Queue()
    max_workers = len(gpus) * max_worker_per_gpu
    for gpu in gpus:
        for _ in range(max_worker_per_gpu):
            gpu_queue.put(gpu)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_predictor,
        initargs=(yolo_model, yolo_kwargs, type, engine, correlate, gpu_queue),
    ) as executor:
        list(
            tqdm(
                executor.map(_multiprocess_predict_correlate_store, videos),
                total=len(videos),
                desc=f"Predicting and correlating {type}",
                unit="video",
            )
        )
