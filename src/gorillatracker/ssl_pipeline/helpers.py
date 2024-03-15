from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Generator

import cv2
from shapely.geometry import Polygon
from sqlalchemy.orm import Session
from sqlalchemy.sql import select

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)


def jenkins_hash(key: int) -> int:
    hash_value = ((key >> 16) ^ key) * 0x45D9F3B
    hash_value = ((hash_value >> 16) ^ hash_value) * 0x45D9F3B
    hash_value = (hash_value >> 16) ^ hash_value & 0xFFFFFFFF
    return hash_value


def video_frame_iterator(
    cap: cv2.VideoCapture, frame_step: int, strict: bool = True
) -> Generator[cv2.typing.MatLike, None, None]:
    frame_nr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_nr % frame_step == 0:
            yield frame
        frame_nr += 1


@contextmanager
def video(
    video_path: Path, frame_step: int, strict: bool = True
) -> Generator[Generator[cv2.typing.MatLike, None, None], None, None]:
    """
    Context manager for reading frames from a video file.

    Args:
        video_path (Path): The path to the video file.
        frame_step (int): The step size for reading frames.
        strict (bool, optional): If True, raises an error if not all frames are consumed. Defaults to True.

    Yields:
        Generator[cv2.typing.MatLike, None, None]: A generator that yields frames from the video.

    Raises:
        ValueError: If not all frames are consumed and strict is True.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield video_frame_iterator(cap, frame_step, strict)
    finally:
        cap.release()


@dataclass(frozen=True)
class TrackedBoundingBox:
    id: int
    bbox: BoundingBox


@dataclass(frozen=True)
class BoundingBox:
    x_center_n: float
    y_center_n: float
    width_n: float
    height_n: float
    confidence: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        assert 0 <= self.x_center_n <= 1, "x_center_n must be in the range [0, 1]"
        assert 0 <= self.y_center_n <= 1, "y_center_n must be in the range [0, 1]"
        assert 0 <= self.width_n <= 1, "width_n must be in the range [0, 1]"
        assert 0 <= self.height_n <= 1, "height_n must be in the range [0, 1]"
        assert 0 <= self.confidence <= 1, "confidence must be in the range [0, 1]"

    def iou(self, other: BoundingBox) -> float:
        return self.polygon.intersection(other.polygon).area / self.polygon.union(other.polygon).area

    @property
    def polygon(self) -> Polygon:
        xtl, ytl = self.top_left
        xbr, ybr = self.bottom_right
        return Polygon([(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)])

    @property
    def x_top_left(self) -> int:
        return int((self.x_center_n - self.width_n / 2) * self.image_width)

    @property
    def y_top_left(self) -> int:
        return int((self.y_center_n - self.height_n / 2) * self.image_height)

    @property
    def x_bottom_right(self) -> int:
        return int((self.x_center_n + self.width_n / 2) * self.image_width)

    @property
    def y_bottom_right(self) -> int:
        return int((self.y_center_n + self.height_n / 2) * self.image_height)

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x_top_left, self.y_top_left

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x_bottom_right, self.y_bottom_right

    @classmethod
    def from_tracking_frame_feature(cls, frame_feature: TrackingFrameFeature) -> BoundingBox:
        return cls(
            frame_feature.bbox_x_center,
            frame_feature.bbox_y_center,
            frame_feature.bbox_width,
            frame_feature.bbox_height,
            frame_feature.confidence,
            frame_feature.tracking.video.width,
            frame_feature.tracking.video.height,
        )


@dataclass(frozen=True)
class TrackedFrame:
    frame_nr: int
    frame_features: list[TrackingFrameFeature]


def get_tracked_frames(session: Session, video: Video, filter_by_type: str | None = None) -> list[TrackedFrame]:
    """
    Retrieves a list of TrackedFrame objects for the given video.

    Args:
        session (Session): The SQLAlchemy session object used to execute database queries.
        video (Video): The Video object representing the video for which to retrieve tracked frames.
        filter_by_type (str | None, optional): Optional filter to retrieve only tracked frames with a specific type.
            If None, all tracked frames are retrieved. Defaults to None.

    Returns:
        list[TrackedFrame]: A list of TrackedFrame objects representing the tracked frames for the given video.
            - Each TrackedFrame object contains the frame number and a list of TrackingFrameFeature objects for that frame.
            - The list of TrackedFrame objects covers all sampled frames of the video, with a frame step determined by video.frame_step.
            - The length of the returned list is equal to ```math.ceil(video.frames / video.frame_step)```.
            - Note: Some TrackedFrame objects may have an empty list of frame_features if no features were tracked for that frame.

    """
    tracked_frames: list[TrackedFrame] = []

    stmt = (
        select(TrackingFrameFeature)
        .join(Tracking)
        .where(Tracking.video_id == video.video_id)
        .order_by(TrackingFrameFeature.frame_nr)
    )
    if filter_by_type:
        stmt = stmt.where(TrackingFrameFeature.type == filter_by_type)

    frame_feature_query = session.scalars(stmt).all()

    frame_features_grouped = {
        frame_nr: list(features) for frame_nr, features in groupby(frame_feature_query, key=lambda x: x.frame_nr)
    }

    for frame_nr in range(0, video.frames, video.frame_step):
        frame_features = frame_features_grouped.get(frame_nr, [])
        tracked_frame = TrackedFrame(frame_nr=frame_nr, frame_features=frame_features)
        tracked_frames.append(tracked_frame)

    assert len(tracked_frames) == math.ceil(video.frames / video.frame_step)
    return tracked_frames
