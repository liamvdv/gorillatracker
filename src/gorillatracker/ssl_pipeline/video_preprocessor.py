from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.models import Camera, Video


class MetadataExtractor(Protocol):
    def __call__(self, video_path: Path) -> VideoMetadata: ...


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
    sampled_fps: int,
    session_cls: sessionmaker[Session],
    metadata_extractor: MetadataExtractor,
) -> None:
    metadata = metadata_extractor(video_path)
    properties = video_properties_extractor(video_path)
    assert properties.fps % sampled_fps == 0, "Sampled FPS must be a factor of the original FPS"
    video = Video(
        path=str(video_path),
        version=version,
        start_time=metadata.start_time,
        width=properties.width,
        height=properties.height,
        fps=properties.fps,
        sampled_fps=sampled_fps,
        frames=properties.frames,
    )

    with session_cls() as session:
        camera = session.execute(select(Camera).where(Camera.name == metadata.camera_name)).scalar_one()
        camera.videos.append(video)
        session.commit()


def preprocess_videos(
    video_paths: list[Path],
    version: str,
    sampled_fps: int,
    engine: Engine,
    metadata_extractor: MetadataExtractor,
) -> None:
    session_cls = sessionmaker(bind=engine)
    assert all(video_path.exists() for video_path in video_paths), "All videos must exist"
    for video_path in tqdm(video_paths, desc="Preprocessing videos"):
        preprocess_and_store(video_path, version, sampled_fps, session_cls, metadata_extractor)
