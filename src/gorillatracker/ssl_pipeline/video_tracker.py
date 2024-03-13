from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.models import Base, Camera, Tracking, TrackingFrameFeature, Video

tracker = None
session_cls = None
metadata_extractor = None
tracker_config = None


@dataclass(frozen=True)
class VideoMetadata:
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
) -> None:
    global tracker, session_cls, metadata_extractor, tracker_config
    metadata_extractor = video_metadata_extractor
    tracker_config = tracker_cfg
    tracker = YOLO(tracker_model)
    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    session_cls = sessionmaker(bind=engine)


def track_and_store(video: Path) -> None:
    global tracker, session_cls, metadata_extractor, tracker_config
    assert tracker is not None, "Tracker is not initialized, call init_tracker first"
    assert session_cls is not None, "Session class is not initialized, call init_tracker first"
    assert metadata_extractor is not None, "Metadata extractor is not initialized, call init_tracker first"
    assert tracker_config is not None, "Tracker config is not initialized, call init_tracker first"
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
    trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=tracked_video))
    result: results.Results
    for relative_frame, result in enumerate(
        tracker.track(
            video,
            stream=True,
            half=True,
            device="cuda:0",
            vid_stride=tracker_config.frame_stride,
            tracker=tracker_config.tracker_config,
            iou=tracker_config.iou,
            conf=tracker_config.conf,
        )
    ):
        # TODO: remove
        if relative_frame > 200:
            break

        detections = result.boxes
        frame = relative_frame * tracker_config.frame_stride
        assert isinstance(detections, results.Boxes)
        for detection in detections:
            assert detection.id is not None
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

    with session_cls.begin() as session:
        stmt = select(Camera).where(Camera.name == metadata.camera_name)
        camera = session.execute(stmt).scalar_one()
        tracked_video.camera = camera
        session.add(tracked_video)


def multiprocess_video_tracker(
    yolo_model: Path,
    videos: list[Path],
    tracker_config: TrackerConfig,
    engine: Engine,
    metadata_extractor: Callable[[Path], VideoMetadata],
    max_workers=4,
    n_gpus=1,
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
    assert len(videos) == set(map(lambda x: x.stem, videos)), "Videos must have unique filenames"
    with Session(engine) as session:
        cameras = session.execute(select(Camera)).scalars().all()
        assert cameras, "No cameras found in the database"
        assert all(
            metadata_extractor(video).camera_name in map(lambda x: x.name, cameras) for video in videos
        ), "All videos must have a corresponding camera in the database"

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_tracker,
        initargs=(yolo_model, engine, metadata_extractor, tracker_config),
    ) as executor:
        tqdm(executor.map(track_and_store, videos), total=len(videos), desc="Tracking videos", unit="video")


# if __name__ == "__main__":
#     os.remove("test.db")
#     engine = create_engine("sqlite:///test.db")
#     benchmark_video = Path("video_data/R108_20230213_301.mp4")
#     Base.metadata.create_all(engine)

#     with Session(engine) as session:
#         session.add(Camera(name="R108", latitude=0, longitude=0))
#         session.commit()

#     with ProcessPoolExecutor(
#         initializer=init_tracker, initargs=(Path("models/yolov8n_gorilla_body.pt"), engine)
#     ) as executor:
#         executor.map(
#             track_and_store,
#             [benchmark_video],
#             [video_metadata_extractor],
#             [TrackerConfig(10, Path("cfgs/tracker/botsort.yaml"))],
#         )

#     with Session(engine) as session:
#         stmt = select(Video).where(Video.filename == "R108_20230213_301.mp4")
#         video = session.execute(stmt).scalar_one()
#         print(video)
#         print(video.camera)
#         print(len(video.trackings))

#     from gorillatracker.ssl_pipeline.visualizer import visualize_video

#     visualize_video(benchmark_video, engine, Path("output.mp4"))
