from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Generator

import cv2
from sqlalchemy.orm import Session
from sqlalchemy.sql import select

from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)


def jenkins_hash(key: int) -> int:
    hash_value = ((key >> 16) ^ key) * 0x45D9F3B
    hash_value = ((hash_value >> 16) ^ hash_value) * 0x45D9F3B
    hash_value = (hash_value >> 16) ^ hash_value & 0xFFFFFFFF
    return hash_value


def video_generator(video: Path, frame_step: int) -> Generator[cv2.typing.MatLike, None, None]:
    cap = cv2.VideoCapture(str(video))
    frame_nr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_nr % frame_step == 0:
            yield frame
        frame_nr += 1
    cap.release()


def assert_generator_exhausted(generator: Generator[Any, None, None]) -> None:
    try:
        next(generator)
        raise AssertionError("Generator was not exhausted")
    except StopIteration:
        pass


@dataclass(frozen=True)
class BoundingBox:
    x_top_left: int
    y_top_left: int
    x_bottom_right: int
    y_bottom_right: int

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x_top_left, self.y_top_left

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x_bottom_right, self.y_bottom_right

    @staticmethod
    def from_tracking_frame_feature(frame_feature: TrackingFrameFeature) -> BoundingBox:
        return BoundingBox.from_yolo(
            frame_feature.bbox_x_center,
            frame_feature.bbox_y_center,
            frame_feature.bbox_width,
            frame_feature.bbox_height,
            frame_feature.tracking.video.width,
            frame_feature.tracking.video.height,
        )

    @staticmethod
    def from_yolo(
        x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
    ) -> BoundingBox:
        x_top_left = int((x_center - width / 2) * image_width)
        y_top_left = int((y_center - height / 2) * image_height)
        x_bottom_right = int((x_center + width / 2) * image_width)
        y_bottom_right = int((y_center + height / 2) * image_height)
        return BoundingBox(x_top_left, y_top_left, x_bottom_right, y_bottom_right)


@dataclass(frozen=True)
class TrackedFrame:
    frame_nr: int
    frame_features: list[TrackingFrameFeature]


def get_tracked_frames(session: Session, video: Video) -> list[TrackedFrame]:
    tracked_frames: list[TrackedFrame] = []

    stmt = (
        select(TrackingFrameFeature)
        .join(Tracking)
        .where(Tracking.video_id == video.video_id)
        .order_by(TrackingFrameFeature.frame_nr)
    )
    frame_feature_query = session.scalars(stmt).all()

    if len(frame_feature_query) < 10:
        log.warning(f"Video {video.filename} has less than 10 frames with tracked features")

    frame_features_grouped = {
        frame_nr: list(features) for frame_nr, features in groupby(frame_feature_query, key=lambda x: x.frame_nr)
    }

    for frame_nr in range(0, video.frames, video.frame_step):
        frame_features = frame_features_grouped.get(frame_nr, [])
        tracked_frame = TrackedFrame(frame_nr=frame_nr, frame_features=frame_features)
        tracked_frames.append(tracked_frame)

    assert len(tracked_frames) == math.ceil(video.frames / video.frame_step)
    return tracked_frames
