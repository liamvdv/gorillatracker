from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Callable

import cv2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.models import Camera, Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)

tracker = None
session_cls = None
metadata_extractor = None
tracker_config = None
assigned_gpu = None


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


@dataclass(frozen=True)
class TrackerConfig:
    frame_stride: int  # sample every nth frame
    tracker_config: Path
    iou = 0.2
    conf = 0.7


def video_properties_extractor(video: Path) -> VideoProperties:
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return VideoProperties(frames, width, height, fps)


def init_tracker(
    tracker_model: Path,
    engine: Engine,
    video_metadata_extractor: Callable[[Path], VideoMetadata],
    tracker_cfg: TrackerConfig,
    gpu_queue: Queue[int],
) -> None:
    global tracker, session_cls, metadata_extractor, tracker_config, assigned_gpu
    metadata_extractor = video_metadata_extractor
    tracker_config = tracker_cfg
    tracker = YOLO(tracker_model)
    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    session_cls = sessionmaker(bind=engine)
    assigned_gpu = gpu_queue.get()
    log.info(f"Tracker initialized on GPU {assigned_gpu}")


def track_and_store(video: Path) -> None:
    log = logging.getLogger(__name__)
    global tracker, session_cls, metadata_extractor, tracker_config, assigned_gpu
    assert tracker is not None, "Tracker is not initialized, call init_tracker first"
    assert session_cls is not None, "Session class is not initialized, use init_tracker instead"
    assert metadata_extractor is not None, "Metadata extractor is not initialized, use init_tracker instead"
    assert tracker_config is not None, "Tracker config is not initialized, use init_tracker instead"
    assert assigned_gpu is not None, "GPU is not assigned, use init_tracker instead"
    metadata = metadata_extractor(video)
    properties = video_properties_extractor(video)
    tracked_video = Video(
        filename=video.name,
        start_time=metadata.start_time,
        width=properties.width,
        height=properties.height,
        fps=properties.fps,
        frames=properties.frames,
    )

    # Create a VideoCapture object using the memory-mapped file

    trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=tracked_video))
    result: results.Results
    for relative_frame, result in enumerate(
        tracker.track(
            video,
            stream=True,
            half=True,
            device=f"cuda:{assigned_gpu}",
            vid_stride=tracker_config.frame_stride,
            tracker=tracker_config.tracker_config,
            iou=tracker_config.iou,
            conf=tracker_config.conf,
            verbose=False,
        )
    ):
        detections = result.boxes
        frame = relative_frame * tracker_config.frame_stride
        assert isinstance(detections, results.Boxes)
        for detection in detections:
            if detection.id is None:
                log.warning(f"For video {video.name}, frame {frame}, no associated tracking id found, continuing...")
                continue
            tracking_id = int(detection.id[0].int().item())
            x, y, w, h = detection.xywhn[0].tolist()
            confidence = detection.conf.item()
            tracking = trackings[tracking_id]
            tracking.frame_features.append(
                TrackingFrameFeature(
                    frame_nr=frame,
                    bbox_x_center=x,
                    bbox_y_center=y,
                    bbox_width=w,
                    bbox_height=h,
                    confidence=confidence,
                    type="body",  # NOTE(memben): Tracking will always be done using the body model
                )
            )

    with session_cls() as session:
        stmt = select(Camera).where(Camera.name == metadata.camera_name)
        camera = session.execute(stmt).scalar_one()
        tracked_video.camera = camera
        session.add(tracked_video)
        session.commit()


def multiprocess_video_tracker(
    yolo_model: Path,
    videos: list[Path],
    tracker_cfg: TrackerConfig,
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
        videos (list[Path]): The videos to track.
        tracker_config (TrackerConfig): The tracker settings to use.
        engine (Engine): The database engine (Note sqlite:///:memory: is not supported).
        metadata_extractor (Callable[[Path], VideoMetadata]): A function to extract the metadata from the video.
        max_workers (int, optional): The number of workers to use for tracking. Defaults to 4.
        n_gpus (int, optional): The number of GPUs to use for tracking. Defaults to 1.

    Returns:
        None, the videos are tracked and stored in the database.
    """
    assert len(videos) == len(set(map(lambda x: x.stem, videos))), "Videos must have unique filenames"
    with Session(engine) as session:
        cameras = session.execute(select(Camera)).scalars().all()
        assert cameras, "No cameras found in the database"
        tracked_videos = session.execute(select(Video.filename)).scalars().all()
        assert not set(map(lambda x: x.name, videos)) & set(tracked_videos), "Some videos are already tracked"
        # TODO(memben): This could be a costly operation (e.g. OCR), consider caching the result of the metadata extractor
        assert all(
            metadata_extractor(video).camera_name in map(lambda x: x.name, cameras) for video in videos
        ), "All videos must have a corresponding camera in the database"

    log.info("Tracking videos...")
    gpu_queue: Queue[int] = Queue()
    max_workers = len(gpus) * max_worker_per_gpu
    for gpu in gpus:
        for _ in range(max_worker_per_gpu):
            gpu_queue.put(gpu)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_tracker,
        initargs=(yolo_model, engine, metadata_extractor, tracker_cfg, gpu_queue),
    ) as executor:
        list(tqdm(executor.map(track_and_store, videos), total=len(videos), desc="Tracking videos", unit="video"))

    with Session(engine) as session:
        stmt = select(Video.filename)
        tracked_videos = session.execute(stmt).scalars().all()
        # NOTE: We can run multiprocessing multiple times
        assert set(map(lambda x: x.name, videos)) - set(tracked_videos) == set(), "Not all videos were tracked"
