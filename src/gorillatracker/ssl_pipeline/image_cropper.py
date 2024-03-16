import logging
import random
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Callable

import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from gorillatracker.ssl_pipeline.helpers import BoundingBox, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import load_video_by_filename, load_video_tracking_frame_features

log = logging.getLogger(__name__)


def group_by_tracking_id(frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
    frame_features.sort(key=lambda x: x.tracking.tracking_id)
    return {
        tracking_id: list(features)
        for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
    }


def random_sampling(
    n_images_per_tracking: int, frames_features: list[TrackingFrameFeature], cutout_feature_types: list[str]
) -> list[TrackingFrameFeature]:
    tracking_id_grouped = group_by_tracking_id(frames_features)
    sampled_features = []
    for features in tracking_id_grouped.values():
        filtered_features = [f for f in features if f.type in cutout_feature_types]
        num_samples = min(len(filtered_features), n_images_per_tracking)
        sampled_features.extend(random.sample(filtered_features, num_samples))
    return sampled_features


def random_sampling_strategy(
    n_images_per_tracking: int, cutout_feature_types: list[str] = ["body"]
) -> Callable[[list[TrackingFrameFeature]], list[TrackingFrameFeature]]:
    """
    Returns a function that performs random sampling of TrackingFrameFeature instances.

    Args:
        n_images_per_tracking: The number of images to sample per tracking ID.
        cutout_feature_types: A list of feature types to include in the sampling (default: ["body"]).

    Returns:
        A function that takes a list of TrackingFrameFeature instances and returns a randomly
        sampled subset based on the specified criteria.
    """
    return partial(random_sampling, n_images_per_tracking, cutout_feature_types=cutout_feature_types)


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
    tracking_frame_feature_filter: Callable[[list[TrackingFrameFeature]], list[TrackingFrameFeature]],
    dest_base_path: Path,
) -> None:
    with session_cls() as session:
        video_tracking = load_video_by_filename(session, video)
        dest_path = dest_base_path / video_tracking.camera.name / video_tracking.filename

        frame_features = list(load_video_tracking_frame_features(session, video_tracking.video_id))
        filtered_frame_features = tracking_frame_feature_filter(frame_features)
        crop_tasks = [
            CropTask(
                feature.frame_nr, destination_path(dest_path, feature), BoundingBox.from_tracking_frame_feature(feature)
            )
            for feature in filtered_frame_features
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


if __name__ == "__main__":
    engine = create_engine("sqlite:///test.db")
    video = Path("video_data/R019_20221228_607.mp4")
    session_cls = sessionmaker(bind=engine)
    crop_from_video(video, session_cls, random_sampling_strategy(3), Path("cropped_images"))
    from gorillatracker.ssl_pipeline.visualizer import visualize_video

    # visualize_video(video, session_cls, Path("visualized.mp4"))
