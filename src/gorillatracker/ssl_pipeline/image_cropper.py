import logging
import random
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import cv2
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.helpers import BoundingBox, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import load_video_by_filename, video_tracking_frame_features_query

log = logging.getLogger(__name__)


def group_by_tracking_id(frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
    frame_features.sort(key=lambda x: x.tracking.tracking_id)
    return {
        tracking_id: list(features)
        for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
    }


class SamplingStrategy(ABC):
    """Defines a sampling strategy for selecting a subset of TrackingFrameFeature instances to crop."""

    def filter(self, video: Path, session: Session) -> list[TrackingFrameFeature]:
        """Returns a subset of TrackingFrameFeature instances to crop, prefiltered by the sampling strategy."""
        frames_features = list(session.execute(self.filter_query(video)).scalars().all())
        return self.filter_features(frames_features)

    def filter_query(self, video: Path) -> Select[tuple[TrackingFrameFeature]]:
        """Filter query to select a subset of TrackingFrameFeature instances on the database."""
        return video_tracking_frame_features_query(video)

    def filter_features(self, frame_features: list[TrackingFrameFeature]) -> list[TrackingFrameFeature]:
        """Defines how to filter a list of TrackingFrameFeature instances."""
        return frame_features


class RandomSampling(SamplingStrategy):
    """Randomly samples a subset of TrackingFrameFeature instances to crop."""

    def __init__(self, n_images_per_tracking: int, cutout_feature_types: list[str] = ["body"]):
        self.n_images_per_tracking = n_images_per_tracking
        self.cutout_feature_types = cutout_feature_types

    def filter_features(self, frame_features: list[TrackingFrameFeature]) -> list[TrackingFrameFeature]:
        tracking_id_grouped = group_by_tracking_id(frame_features)
        sampled_features = []
        for features in tracking_id_grouped.values():
            filtered_features = [f for f in features if f.type in self.cutout_feature_types]
            num_samples = min(len(filtered_features), self.n_images_per_tracking)
            sampled_features.extend(random.sample(filtered_features, num_samples))
        return sampled_features


@dataclass(frozen=True, order=True)
class CropTask:
    frame_nr: int
    dest: Path
    bounding_box: BoundingBox = field(compare=False)


def destination_path(base_path: Path, feature: TrackingFrameFeature) -> Path:
    return Path(base_path, str(feature.tracking.tracking_id), f"{feature.frame_nr}.png")


def crop_from_video(
    video: Path,
    session_cls: sessionmaker[Session],
    sampling_strategy: SamplingStrategy,
    dest_base_path: Path,
) -> None:
    with session_cls() as session:
        video_tracking = load_video_by_filename(session, video)
        dest_path = dest_base_path / video_tracking.camera.name / video_tracking.filename
        frame_features = sampling_strategy.filter(video, session)
        crop_tasks = [
            CropTask(
                feature.frame_nr, destination_path(dest_path, feature), BoundingBox.from_tracking_frame_feature(feature)
            )
            for feature in frame_features
        ]

    crop_queue = deque(sorted(crop_tasks))

    if not crop_queue:
        log.warning(f"No frames to crop for video: {video}")
        return

    with video_reader(video) as video_feed:
        for video_frame in video_feed:
            while crop_queue and video_frame.frame_nr == crop_queue[0].frame_nr:
                crop_task = crop_queue.popleft()
                cropped_frame = video_frame.frame[
                    crop_task.bounding_box.y_top_left : crop_task.bounding_box.y_bottom_right,
                    crop_task.bounding_box.x_top_left : crop_task.bounding_box.x_bottom_right,
                ]
                crop_task.dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_task.dest), cropped_frame)
        assert not crop_queue, "Not all crop tasks were completed"