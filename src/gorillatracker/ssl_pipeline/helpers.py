from __future__ import annotations

import json
import logging
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time
from itertools import groupby
from pathlib import Path
from typing import Generator, Iterator, Optional, Sequence

import cv2
import easyocr
from shapely.geometry import Polygon

from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video

log = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    frame_nr: int
    frame: cv2.typing.MatLike


def jenkins_hash(key: int) -> int:
    hash_value = ((key >> 16) ^ key) * 0x45D9F3B
    hash_value = ((hash_value >> 16) ^ hash_value) * 0x45D9F3B
    hash_value = (hash_value >> 16) ^ hash_value & 0xFFFFFFFF
    return hash_value


def video_frame_iterator(cap: cv2.VideoCapture, frame_step: int) -> Generator[VideoFrame, None, None]:
    frame_nr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_nr % frame_step == 0:
            yield VideoFrame(frame_nr, frame)
        frame_nr += 1


@contextmanager
def video_reader(video_path: Path, frame_step: int = 1) -> Iterator[Generator[VideoFrame, None, None]]:
    """
    Context manager for reading frames from a video file.

    Args:
        video_path (Path): The path to the video file.
        frame_step (int): The step size for reading frames.


    Yields:
        Generator[VideoFrame, None, None]: A generator that yields VideoFrame objects.

    """
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Could not open video file: {video_path}"
    try:
        yield video_frame_iterator(cap, frame_step)
    finally:
        cap.release()


@dataclass(frozen=True)
class AssociatedBoundingBox:
    association: int
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

    def intersection_over_smallest_area(self, other: BoundingBox) -> float:
        intersection = self.polygon.intersection(other.polygon)
        return intersection.area / min(self.polygon.area, other.polygon.area)

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
            frame_feature.bbox_x_center_n,
            frame_feature.bbox_y_center_n,
            frame_feature.bbox_width_n,
            frame_feature.bbox_height_n,
            frame_feature.confidence,
            int(frame_feature.bbox_width / frame_feature.bbox_width_n),
            int(frame_feature.bbox_height / frame_feature.bbox_height_n),
        )


def groupby_frame(
    tracking_frame_features: Sequence[TrackingFrameFeature],
) -> defaultdict[int, list[TrackingFrameFeature]]:
    sorted_features = sorted(tracking_frame_features, key=lambda x: x.frame_nr)
    frame_features = defaultdict(list)
    for frame_nr, features in groupby(sorted_features, key=lambda x: x.frame_nr):
        frame_features[frame_nr] = list(features)
    return frame_features


def remove_processed_videos(video_paths: list[Path], processed_videos: list[Video]) -> list[Path]:
    processed_video_paths = [Path(v.path) for v in processed_videos]
    return [v for v in video_paths if v not in processed_video_paths]


def crop_frame(frame: cv2.typing.MatLike, bbox: BoundingBox) -> cv2.typing.MatLike:
    """Crops a frame according to the bounding box."""
    cropped_frame = frame[
        bbox.y_top_left : bbox.y_bottom_right,
        bbox.x_top_left : bbox.x_bottom_right,
    ]
    return cropped_frame


def read_timestamp(video_path: Path, bbox: BoundingBox, ocr_reader: Optional[easyocr.Reader] = None) -> time:
    """
    Extracts the time stamp from the video file.

    Args:
        video_path (Path): path to the video file
        bbox (BoundingBox): bounding box for the time stamp
        ocr_reader (Optional[easyocr.Reader]): OCR reader

    Returns:
        time: time stamp as time object
    """
    with video_reader(video_path) as video_feed:
        frame = next(video_feed).frame

    cropped_frame = crop_frame(frame, bbox)
    ocr_reader = ocr_reader or easyocr.Reader(["en"], gpu=False, verbose=False)
    time_stamp = _extract_time_stamp(cropped_frame, ocr_reader)

    return time_stamp


def _extract_time_stamp(cropped_frame: cv2.typing.MatLike, reader: easyocr.Reader) -> time:
    """Extracts the time stamp from the cropped frame."""
    TIMESTAMP_CHARS = "0123456789APM:"
    rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    extracted_time_stamp_raw = reader.readtext(rgb_frame, allowlist=TIMESTAMP_CHARS)
    time_stamp = "".join(text[1] for text in extracted_time_stamp_raw).replace(":", "")
    return _extract_time(time_stamp)


def _extract_time(time_stamp: str) -> time:
    try:
        return datetime.strptime(time_stamp, "%I%M%p").time()
    except ValueError:
        raise ValueError("Could not extract time stamp from frame")


def extract_meta_data_time(video_path: Path) -> Optional[datetime]:
    """
    Extracts the creation time of the video from the metadata.
    """
    time_stamp = _extract_iso_timestamp(video_path)
    if time_stamp is None:
        return None
    return datetime.fromisoformat(time_stamp.replace("Z", "+00:00"))


def _extract_iso_timestamp(video_path: Path) -> Optional[str]:
    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time",
        video_path,
    ]

    try:
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)  # type: ignore
        metadata = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"ffprobe execution failed: {e.stderr}")
        raise
    except json.JSONDecodeError:
        log.error("Failed to decode JSON from ffprobe output.")
        raise

    try:
        creation_time = metadata["format"]["tags"]["creation_time"]
        return creation_time
    except KeyError:
        log.error(f"Creation time not found in video metadata for {video_path}")
        return None
