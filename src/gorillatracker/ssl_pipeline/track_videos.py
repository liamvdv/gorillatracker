import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from turtle import width
from typing import Callable

import cv2
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from ultralytics import YOLO
from ultralytics.engine import results

from gorillatracker.ssl_pipeline.models import Base, Camera, Tracking, TrackingFrameFeature, Video

tracker = None
session_cls = None


# Use case specific
def video_camera_name_extractor(video: Path) -> str:
    return video.stem.split("_")[0]


# Use case specific
def video_start_time_extractor(video: Path) -> datetime:
    _, date_str, _ = video.stem.split("_")
    timestamps_path = "data/derived_data/timestamps.json"
    with open(timestamps_path, "r") as f:
        timestamps = json.load(f)
    date = datetime.strptime(date_str, "%Y%m%d")
    timestamp = timestamps[video.stem]
    daytime = datetime.strptime(timestamp, "%I:%M %p")
    date = datetime.combine(date, daytime.time())
    return date


@dataclass(frozen=True)
class VideoMetadata:
    camera_name: str
    start_time: datetime
    frames: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class TrackerConfig:
    frame_stride: int
    tracker_config: Path


def video_metadata_extractor(video: Path) -> VideoMetadata:
    camera_name = video_camera_name_extractor(video)
    start_time = video_start_time_extractor(video)
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return VideoMetadata(camera_name, start_time, frames, width, height, fps)


def init_tracker(tracker_model: Path, engine: Engine) -> None:
    global tracker, session_cls
    tracker = YOLO(tracker_model)
    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    session_cls = sessionmaker(bind=engine)


def track_and_store(
    video: Path, metadata_extractor: Callable[[Path], VideoMetadata], tracker_settings: TrackerConfig
) -> None:
    global tracker, session_cls
    assert tracker is not None, "Tracker is not initialized, call init_tracker first"
    assert session_cls is not None, "Session class is not initialized, call init_tracker first"
    metadata = metadata_extractor(video)
    tracked_video = Video(
        filename=video.name,
        start_time=metadata.start_time,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        frames=metadata.frames,
    )
    trackings: defaultdict[int, Tracking] = defaultdict(lambda: Tracking(video=tracked_video))
    result: results.Results
    for relative_frame, result in enumerate(
        tracker.track(
            video,
            stream=True,
            vid_stride=tracker_settings.frame_stride,
            device="cuda:0",
            tracker=tracker_settings.tracker_config,
            half=True,
        )
    ):
        detections = result.boxes
        frame = relative_frame * tracker_settings.frame_stride
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
        print(camera)
        tracked_video.camera = camera
        session.add(tracked_video)


if __name__ == "__main__":
    # os.remove("test.db")
    engine = create_engine("sqlite:///test.db", echo=True)
    benchmark_video = Path("video_data/R108_20230213_301.mp4")
    # Base.metadata.create_all(engine)
    # with Session(engine) as session:
    #     session.add(Camera(name="R108", latitude=0, longitude=0))
    #     session.commit()

    # with ProcessPoolExecutor(
    #     initializer=init_tracker, initargs=(Path("models/yolov8n_gorilla_body.pt"), engine)
    # ) as executor:
    #     executor.map(
    #         track_and_store,
    #         [benchmark_video],
    #         [video_metadata_extractor],
    #         [TrackerConfig(5, Path("cfgs/tracker/botsort.yaml"))],
    #     )

    # with Session(engine) as session:
    #     stmt = select(Video).where(Video.filename == "R108_20230213_301.mp4")
    #     video = session.execute(stmt).scalar_one()
    #     print(video)
    #     print(video.camera)
    #     print(len(video.trackings))

    from gorillatracker.ssl_pipeline.visualizer import visualize_video

    visualize_video(benchmark_video, engine, Path("output.mp4"))
