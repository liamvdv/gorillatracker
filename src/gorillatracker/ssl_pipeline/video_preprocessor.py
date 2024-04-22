from __future__ import annotations

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


class MetadataExtractor(Protocol):
    def __call__(self, video_path: Path) -> VideoMetadata: ...


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
) -> None:
    metadata = metadata_extractor(video_path)
    properties = video_properties_extractor(video_path)
    assert properties.fps != 0, "fps should not be 0"
    video = Video(
        path=str(video_path),
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
        session.commit()


def preprocess_videos(
    video_paths: list[Path],
    version: str,
    target_output_fps: int,
    engine: Engine,
    metadata_extractor: MetadataExtractor,
) -> list[Path]:
    """preprocess_videos and return valid video paths."""
    valid_video_paths = video_paths.copy()
    session_cls = sessionmaker(bind=engine)
    for video_path in tqdm(video_paths, desc="Preprocessing videos"):
        try:
            assert video_path.exists(), f"Video {video_path} does not exist"
            preprocess_and_store(video_path, version, target_output_fps, session_cls, metadata_extractor)
        except AssertionError as e:
            print(f"Error processing video {video_path}: {e}")
            valid_video_paths.remove(video_path)
    if len(valid_video_paths) < len(video_paths):
        print(f"Only {len(valid_video_paths)} out of {len(video_paths)} videos were processed.")
    return valid_video_paths
