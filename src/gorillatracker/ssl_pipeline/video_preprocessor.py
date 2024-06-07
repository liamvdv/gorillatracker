from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

import cv2
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.models import Video
from gorillatracker.ssl_pipeline.queries import get_or_create_camera

log = logging.getLogger(__name__)


class MetadataExtractor(Protocol):
    def __call__(self, video_path: Path) -> VideoMetadata: ...


class InsertHook(Protocol):
    def __call__(self, video: Video) -> None: ...


@dataclass(frozen=True)
class VideoMetadata:
    """High level metadata about a video."""

    camera_name: str
    start_time: Optional[datetime]


@dataclass(frozen=True)
class VideoProperties:
    frames: int
    width: int
    height: int
    fps: int


def video_properties_extractor(video_path: Path) -> VideoProperties:
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return VideoProperties(frames, width, height, fps)


def preprocess_and_store(
    video_path: Path,
    version: str,
    target_output_fps: int,
    session_cls: sessionmaker[Session],
    metadata_extractor: MetadataExtractor,
    video_insert_hook: InsertHook,
) -> None:
    try:
        metadata = metadata_extractor(video_path)
        properties = video_properties_extractor(video_path)
    except Exception as e:
        log.warning(f"Failed to extract metadata from video {video_path}: {e}")
        return

    if properties.fps < 1:
        log.warning(f"Video {video_path} has an invalid FPS of {properties.fps}, skipping")
        return

    if properties.frames < 1:
        log.warning(f"Video {video_path} has an invalid number of frames {properties.frames}, skipping")
        return

    video = Video(
        absolute_path=str(video_path),
        version=version,
        start_time=metadata.start_time,
        width=properties.width,
        height=properties.height,
        fps=properties.fps,
        target_output_fps=target_output_fps,
        frames=properties.frames,
    )

    with session_cls() as session:
        camera = get_or_create_camera(session, metadata.camera_name)
        camera.videos.append(video)
        video_insert_hook(video)
        session.commit()


def preprocess_videos(
    video_paths: list[Path],
    version: str,
    target_output_fps: int,
    engine: Engine,
    metadata_extractor: MetadataExtractor,
    video_insert_hook: InsertHook,
) -> None:
    session_cls = sessionmaker(bind=engine)
    assert all(video_path.exists() for video_path in video_paths), "All videos must exist"
    for video_path in tqdm(video_paths, desc="Preprocessing videos"):
        preprocess_and_store(video_path, version, target_output_fps, session_cls, metadata_extractor, video_insert_hook)
