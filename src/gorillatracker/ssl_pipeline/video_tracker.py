from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Callable

import cv2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.models import Camera, Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """High level metadata about a video."""

    camera_name: str
    start_time: datetime


@dataclass(frozen=True)
class VideoProperties:
    frames: int
    width: int
    height: int
    fps: int


def video_properties_extractor(video: Path) -> VideoProperties:
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return VideoProperties(frames, width, height, fps)


def track_and_store(
    video: Path,
    yolo_model: YOLO,
    yolo_kwargs: dict[str, Any],
    session_cls: sessionmaker[Session],
    metadata_extractor: Callable[[Path], VideoMetadata],
    tracker_config: Path,
) -> None:
    vid_stride = yolo_kwargs.get("vid_stride", 1)
    metadata = metadata_extractor(video)
    properties = video_properties_extractor(video)
    assert properties.fps % vid_stride == 0, "vid_stride must be a factor of the original fps"
    tracked_video = Video(
        filename=video.name,
        start_time=metadata.start_time,
        width=properties.width,
        height=properties.height,
        fps=properties.fps,
        sampled_fps=properties.fps // vid_stride,
        frames=properties.frames,
    )

    trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=tracked_video))
    result: results.Results
    for relative_frame, result in enumerate(
        yolo_model.track(
            video,
            stream=True,
            tracker=tracker_config,
            **yolo_kwargs,
        )
    ):
        detections = result.boxes
        frame = relative_frame * vid_stride
        assert isinstance(detections, results.Boxes)
        for detection in detections:
            if detection.id is None:
                log.warning(f"For video {video.name}, frame {frame}, no associated tracking id found, continuing...")
                continue
            tracking_id = int(detection.id[0].int().item())
            x, y, w, h = detection.xywhn[0].tolist()
            confidence = detection.conf.item()
            tracking = trackings[tracking_id]
            TrackingFrameFeature(
                tracking=tracking,
                frame_nr=frame,
                bbox_x_center=x,
                bbox_y_center=y,
                bbox_width=w,
                bbox_height=h,
                confidence=confidence,
                type="body",  # NOTE(memben): Tracking will always be done using the body model
            )

    with session_cls() as session:
        camera = session.execute(select(Camera).where(Camera.name == metadata.camera_name)).scalar_one()
        camera.videos.append(tracked_video)
        session.commit()


_yolo_model = None
_yolo_kwargs = None
_session_cls = None
_metadata_extractor = None
_tracker_config = None


def _init_tracker(
    yolo_model: Path,
    yolo_kwargs: dict[str, Any],
    engine: Engine,
    metadata_extractor: Callable[[Path], VideoMetadata],
    tracker_config: Path,
    gpu_queue: Queue[int],
) -> None:
    log = logging.getLogger(__name__)
    global _yolo_model, _yolo_kwargs, _session_cls, _metadata_extractor, _tracker_config
    _yolo_model = YOLO(yolo_model)
    _yolo_kwargs = yolo_kwargs
    _metadata_extractor = metadata_extractor
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


def _multiprocess_track_and_store(video: Path) -> None:
    global _yolo_model, _yolo_kwargs, _session_cls, _metadata_extractor, _tracker_config
    assert _yolo_model is not None, "Tracker is not initialized, call init_tracker first"
    assert _yolo_kwargs is not None, "YOLO kwargs are not initialized, use init_tracker instead"
    assert _session_cls is not None, "Session class is not initialized, use init_tracker instead"
    assert _metadata_extractor is not None, "Metadata extractor is not initialized, use init_tracker instead"
    assert _tracker_config is not None, "Tracker config is not initialized, use init_tracker instead"
    track_and_store(video, _yolo_model, _yolo_kwargs, _session_cls, _metadata_extractor, _tracker_config)


def multiprocess_video_tracker(
    yolo_model: Path,
    yolo_kwargs: dict[str, Any],
    videos: list[Path],
    tracker_config: Path,
    metadata_extractor: Callable[[Path], VideoMetadata],
    engine: Engine,
    max_worker_per_gpu: int = 8,
    gpus: list[int] = [0],
) -> None:
    """
    Track and store the videos in the database using the YOLO model and the tracker settings.
    The database should already contain the cameras that the videos are from.

    Args:
        yolo_model (Path): The YOLO model to use for tracking (single_cls, pre-trained on the body of the animal).
        yolo_kwargs (dict[str, Any]): The keyword arguments to pass to the YOLO model.
        videos (list[Path]): The videos to track.
        tracker_config (Path): The tracker settings to use.
        metadata_extractor (Callable[[Path], VideoMetadata]): A function to extract the metadata from the video.
        engine (Engine): The database engine (Note sqlite:///:memory: is not supported).
        max_worker_per_gpu (int, optional): The maximum number of workers to spawn per GPU. Defaults to 8.
        gpus (list[int], optional): The GPUs to use for tracking. Defaults to [0].

    Returns:
        None, the videos are tracked and stored in the database.
    """
    assert len(videos) == len(set(map(lambda x: x.stem, videos))), "Videos must have unique filenames"
    with Session(engine) as session:
        tracked_videos = session.execute(select(Video.filename)).scalars().all()
        assert not set(map(lambda x: x.name, videos)) & set(tracked_videos), "Some videos are already tracked"
        cameras = session.execute(select(Camera)).scalars().all()
        assert cameras, "No cameras found in the database"
        assert all(
            metadata_extractor(video).camera_name in map(lambda x: x.name, cameras) for video in videos
        ), "All videos must have a corresponding camera in the database"
        # TODO(memben): This could be a costly operation (e.g. OCR), consider caching the result of the metadata extractor

    log.info("Tracking videos...")
    gpu_queue: Queue[int] = Queue()
    max_workers = len(gpus) * max_worker_per_gpu
    for gpu in gpus:
        for _ in range(max_worker_per_gpu):
            gpu_queue.put(gpu)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_tracker,
        initargs=(yolo_model, yolo_kwargs, engine, metadata_extractor, tracker_config, gpu_queue),
    ) as executor:
        list(
            tqdm(
                executor.map(_multiprocess_track_and_store, videos),
                total=len(videos),
                desc="Tracking videos",
                unit="video",
            )
        )

    with Session(engine) as session:
        stmt = select(Video.filename)
        tracked_videos = session.execute(stmt).scalars().all()
        # NOTE: We can run multiprocessing multiple times
        assert set(map(lambda x: x.name, videos)) - set(tracked_videos) == set(), "Not all videos were tracked"
