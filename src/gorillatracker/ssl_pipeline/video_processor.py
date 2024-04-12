from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Literal, Optional

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.helpers import video_reader
from gorillatracker.ssl_pipeline.models import ProcessedVideoFrameFeature, Tracking, TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import load_video

log = logging.getLogger(__name__)


def process_detection(
    box: results.Boxes, video: Video, tracking: Optional[Tracking], frame_nr: int, type: str, session: Session
) -> None:
    x, y, w, h = box.xywhn[0].tolist()
    confidence = box.conf.item()
    session.add(
        TrackingFrameFeature(
            tracking=tracking,
            video=video,
            frame_nr=frame_nr,
            bbox_x_center=x,
            bbox_y_center=y,
            bbox_width=w,
            bbox_height=h,
            confidence=confidence,
            type=type,
        )
    )


def process_prediction(prediction: results.Results, video: Video, frame_nr: int, type: str, session: Session) -> None:
    """Process the prediction and add the tracking frame features to the video, does not commit the session."""

    assert isinstance(prediction.boxes, results.Boxes)
    for detection in prediction.boxes:
        process_detection(detection, video, None, frame_nr, type, session)


def process_tracking(
    prediction: results.Results,
    video: Video,
    trackings: defaultdict[int, Tracking],
    frame_nr: int,
    type: str,
    session: Session,
) -> None:
    """Process the prediction and add the tracking frame features to the video, does not commit the session."""
    assert isinstance(prediction.boxes, results.Boxes)
    for detection in prediction.boxes:
        # NOTE(memben): sometimes the ID is None, this is a bug in the tracker
        tracking_id = int(detection.id[0].int().item()) if detection.id is not None else None
        tracking = trackings[tracking_id] if tracking_id is not None else None
        process_detection(detection, video, tracking, frame_nr, type, session)


def predict_and_store(
    video_path: Path,
    version: str,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    type: str,
    session_cls: sessionmaker[Session],
) -> None:
    with session_cls() as session:
        video = load_video(session, video_path, version)

        with video_reader(video_path, frame_step=video.frame_step) as video_feed:
            for video_frame in video_feed:
                predictions: list[results.Results] = yolo_model.predict(video_frame.frame, **yolo_kwargs)
                assert len(predictions) == 1
                process_prediction(predictions[0], video, video_frame.frame_nr, type, session)

        if len(video.tracking_frame_features) < 10:
            log.warning(f"Video {video_path.name} has less than 10 tracking frame features for {type}")
        video.processed_video_frame_features.append(ProcessedVideoFrameFeature(type=type))
        session.commit()


def track_and_store(
    video_path: Path,
    version: str,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    session_cls: sessionmaker[Session],
    tracker_config: Path,
    type: str,
) -> None:
    with session_cls() as session:
        video = load_video(session, video_path, version)

        trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=video))

        with video_reader(video_path, frame_step=video.frame_step) as video_feed:
            for video_frame in video_feed:
                predictions: list[results.Results] = yolo_model.track(
                    video_frame.frame, tracker=tracker_config, **yolo_kwargs, persist=True
                )
                assert len(predictions) == 1
                process_tracking(predictions[0], video, trackings, video_frame.frame_nr, type, session)

        if len(video.tracking_frame_features) < 10:
            log.warning(f"Video {video_path.name} has less than 10 tracking frame features for {type}")
        video.processed_video_frame_features.append(ProcessedVideoFrameFeature(type=type))
        session.commit()


_version = None
_mode = None
_type = None
_yolo_model = None
_yolo_kwargs = None
_session_cls = None
_tracker_config = None


def _init_processor(
    version: str,
    mode: Literal["tracking", "prediction"],
    type: str,
    yolo_model: Path,
    yolo_kwargs: dict[str, Any],
    engine: Engine,
    tracker_config: Optional[Path],
    gpu_queue: Queue[int],
) -> None:
    log = logging.getLogger(__name__)
    global _version, _mode, _type, _yolo_model, _yolo_kwargs, _session_cls, _tracker_config
    _version = version
    _mode = mode
    _type = type
    _yolo_model = YOLO(yolo_model)
    _yolo_kwargs = yolo_kwargs
    _tracker_config = tracker_config

    assigned_gpu = gpu_queue.get()
    log.info(f"Tracker initialized on GPU {assigned_gpu}")
    if "device" in yolo_kwargs:
        raise ValueError("device will be overwritten by the assigned GPU")
    yolo_kwargs["device"] = f"cuda:{assigned_gpu}"

    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    _session_cls = sessionmaker(bind=engine)


def _multiprocess_process_and_store(video_path: Path) -> None:
    global _version, _mode, _type, _yolo_model, _yolo_kwargs, _session_cls, _tracker_config
    assert _version is not None, "Version is not initialized, call init_processor first"
    assert _mode is not None, "Mode is not initialized, call init_processor first"
    assert _type is not None, "Type is not initialized, call init_processor first"
    assert _yolo_model is not None, "YOLO model is not initialized, call init_processor first"
    assert _yolo_kwargs is not None, "YOLO kwargs are not initialized, use init_processor instead"
    assert _session_cls is not None, "Session class is not initialized, use init_processor instead"
    if _mode == "prediction":
        predict_and_store(video_path, _version, _yolo_model, _yolo_kwargs, _type, _session_cls)
    elif _mode == "tracking":
        assert _tracker_config is not None, "Tracker config is not initialized, use init_processor instead"
        track_and_store(video_path, _version, _yolo_model, _yolo_kwargs, _session_cls, _tracker_config, _type)
    else:
        raise ValueError(f"Mode {_mode} is not supported")


def _multiprocess_video_processor(
    version: str,
    mode: Literal["tracking", "prediction"],
    type: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    video_paths: list[Path],
    engine: Engine,
    max_worker_per_gpu: int = 8,
    tracker_config: Optional[Path] = None,
    gpus: list[int] = [0],
) -> None:
    with Session(engine) as session:
        assert all(video_path.exists() for video_path in video_paths), "Some videos do not exist"
        videos = [load_video(session, video_path, version) for video_path in video_paths]
        assert all(
            processed_video_frame_feature.type != type
            for video in videos
            for processed_video_frame_feature in video.processed_video_frame_features
        ), "Some videos are already processed"

    gpu_queue: Queue[int] = Queue()
    max_workers = len(gpus) * max_worker_per_gpu
    for gpu in gpus:
        for _ in range(max_worker_per_gpu):
            gpu_queue.put(gpu)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_processor,
        initargs=(version, mode, type, yolo_model_path, yolo_kwargs, engine, tracker_config, gpu_queue),
    ) as executor:
        list(
            tqdm(
                executor.map(_multiprocess_process_and_store, video_paths),
                total=len(video_paths),
                desc="Processing videos",
                unit="video",
            )
        )


def multiprocess_track_and_store(
    version: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    video_paths: list[Path],
    tracker_config: Path,
    engine: Engine,
    type: str,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    _multiprocess_video_processor(
        version,
        "tracking",
        type,
        yolo_model_path,
        yolo_kwargs,
        video_paths,
        engine,
        max_worker_per_gpu,
        tracker_config,
        gpus,
    )


def multiprocess_predict_and_store(
    version: str,
    yolo_model_path: Path,
    yolo_kwargs: dict[str, Any],
    video_paths: list[Path],
    engine: Engine,
    type: str,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    _multiprocess_video_processor(
        version, "prediction", type, yolo_model_path, yolo_kwargs, video_paths, engine, max_worker_per_gpu, gpus=gpus
    )
