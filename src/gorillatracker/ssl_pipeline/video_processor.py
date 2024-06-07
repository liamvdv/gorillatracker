from __future__ import annotations

import datetime as dt
import logging
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Engine
from sqlalchemy.orm import Session
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.helpers import video_reader
from gorillatracker.ssl_pipeline.models import TaskType, Tracking, TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import get_next_task, transactional_task

log = logging.getLogger(__name__)


def process_detection(
    box: results.Boxes, video: Video, frame_nr: int, feature_type: str, tracking: Optional[Tracking] = None
) -> TrackingFrameFeature:
    x_n, y_n, w_n, h_n = box.xywhn[0].tolist()
    _, _, w, h = box.xywh[0].tolist()
    confidence = box.conf.item()
    return TrackingFrameFeature(
        tracking=tracking,
        video=video,
        frame_nr=frame_nr,
        bbox_x_center_n=x_n,
        bbox_y_center_n=y_n,
        bbox_width_n=w_n,
        bbox_height_n=h_n,
        bbox_width=w,
        bbox_height=h,
        confidence=confidence,
        feature_type=feature_type,
    )


def process_prediction(
    prediction: results.Results,
    video: Video,
    frame_nr: int,
    feature_type: str,
) -> list[TrackingFrameFeature]:
    assert isinstance(prediction.boxes, results.Boxes)
    return [process_detection(box, video, frame_nr, feature_type) for box in prediction.boxes]


def process_tracking(
    prediction: results.Results,
    video: Video,
    frame_nr: int,
    feature_type: str,
    trackings: defaultdict[int, Tracking],
) -> list[TrackingFrameFeature]:
    assert isinstance(prediction.boxes, results.Boxes)
    detections = []
    for detection in prediction.boxes:
        # NOTE(memben): sometimes the ID is None, this is a bug in the tracker
        tracking_id = int(detection.id[0].int().item()) if detection.id is not None else None
        tracking = trackings[tracking_id] if tracking_id is not None else None
        detections.append(process_detection(detection, video, frame_nr, feature_type, tracking))
    return detections


def predict_and_update(
    session: Session,
    video: Video,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    feature_type: str,
) -> None:
    with video_reader(video.path, frame_step=video.frame_step) as video_feed:
        for video_frame in video_feed:
            predictions: list[results.Results] = yolo_model(video_frame.frame, **yolo_kwargs)
            assert len(predictions) == 1
            session.add_all(process_prediction(predictions[0], video, video_frame.frame_nr, feature_type))


def track_and_update(
    session: Session,
    video: Video,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    tracker_config: Path,
    feature_type: str,
) -> None:
    trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=video))
    with video_reader(video.path, frame_step=video.frame_step) as video_feed:
        for video_frame in video_feed:
            predictions: list[results.Results] = yolo_model.track(
                video_frame.frame, tracker=tracker_config, **yolo_kwargs, persist=True
            )
            assert len(predictions) == 1
            session.add_all(process_tracking(predictions[0], video, video_frame.frame_nr, feature_type, trackings))


def track_worker(
    feature_type: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    engine: Engine,
    tracker_config: Path,
    gpu: int,
) -> None:
    yolo_model = YOLO(yolo_model_path)
    if "device" in yolo_kwargs:
        raise ValueError("device will be overwritten by the assigned GPU")
    yolo_kwargs["device"] = f"cuda:{gpu}"

    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    engine.dispose(close=False)

    with Session(engine) as session:
        for task in get_next_task(
            session, TaskType.TRACK, task_subtype=feature_type, max_retries=3, task_timeout=dt.timedelta(hours=1)
        ):
            with transactional_task(session, task):
                video = task.video
                track_and_update(session, video, yolo_model, yolo_kwargs, tracker_config, feature_type)


def predict_worker(
    feature_type: str, yolo_model_path: Path, yolo_kwargs: dict[str, Any], engine: Engine, gpu: int
) -> None:
    yolo_model = YOLO(yolo_model_path)
    if "device" in yolo_kwargs:
        raise ValueError("device will be overwritten by the assigned GPU")
    yolo_kwargs["device"] = f"cuda:{gpu}"

    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    engine.dispose(close=False)

    with Session(engine) as session:
        for task in get_next_task(
            session, TaskType.PREDICT, task_subtype=feature_type, max_retries=1, task_timeout=dt.timedelta(hours=1)
        ):
            with transactional_task(session, task):
                video = task.video
                predict_and_update(session, video, yolo_model, yolo_kwargs, feature_type)


def multiprocess_track(
    feature_type: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    tracker_config: Path,
    engine: Engine,
    max_worker_per_gpu: int = 8,
    gpu_ids: list[int] = [0],
) -> None:
    gpus = gpu_ids * max_worker_per_gpu

    processes = []
    for gpu in gpus:
        process = multiprocessing.Process(
            target=track_worker, args=(feature_type, yolo_model_path, yolo_kwargs, engine, tracker_config, gpu)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    log.info(f"Tracking {feature_type} completed")


def multiprocess_predict(
    feature_type: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    engine: Engine,
    max_worker_per_gpu: int = 8,
    gpu_ids: list[int] = [0],
) -> None:
    gpus = gpu_ids * max_worker_per_gpu

    processes: list[multiprocessing.Process] = []
    for gpu in gpus:
        process = multiprocessing.Process(
            target=predict_worker, args=(feature_type, yolo_model_path, yolo_kwargs, engine, gpu)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    log.info(f"Prediction {feature_type} completed")
