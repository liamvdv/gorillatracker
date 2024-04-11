from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterator

import cv2
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker

from gorillatracker.ssl_pipeline.dataset import GorillaDataset
from gorillatracker.ssl_pipeline.helpers import BoundingBox, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import load_video

log = logging.getLogger(__name__)


class Sampler:
    """Defines how to sample TrackingFrameFeature instances from the database."""

    def __init__(self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]]) -> None:
        self.query_builder = query_builder

    def sample(self, video_id: int, session: Session) -> Iterator[TrackingFrameFeature]:
        """Sample a subset of TrackingFrameFeature instances from the database. Defined by query and sampling strategy."""

        query = self.query_builder(video_id)
        return iter(session.execute(query).scalars().all())

    def group_by_tracking_id(self, frame_features: list[TrackingFrameFeature]) -> dict[int, list[TrackingFrameFeature]]:
        frame_features.sort(key=lambda x: x.tracking.tracking_id)
        return {
            tracking_id: list(features)
            for tracking_id, features in groupby(frame_features, key=lambda x: x.tracking.tracking_id)
        }


class RandomSampler(Sampler):
    """Randomly sample a subset of TrackingFrameFeature instances per tracking."""

    def __init__(
        self, query_builder: Callable[[int], Select[tuple[TrackingFrameFeature]]], n_samples: int, seed: int = 42
    ) -> None:
        super().__init__(query_builder)
        self.seed = seed
        self.n_samples = n_samples

    def sample(self, video_id: int, session: Session) -> Iterator[TrackingFrameFeature]:
        query = self.query_builder(video_id)
        tracking_frame_features = list(session.execute(query).scalars().all())
        tracking_id_grouped = self.group_by_tracking_id(tracking_frame_features)
        random.seed(self.seed)
        for features in tracking_id_grouped.values():
            num_samples = min(len(features), self.n_samples)
            yield from random.sample(features, num_samples)


@dataclass(frozen=True, order=True)
class CropTask:
    frame_nr: int
    dest: Path
    bounding_box: BoundingBox = field(compare=False)


def destination_path(base_path: Path, feature: TrackingFrameFeature) -> Path:
    return Path(base_path, str(feature.tracking.tracking_id), f"{feature.frame_nr}.png")


def create_crop_tasks(
    video_path: Path,
    version: str,
    sampler: Sampler,
    session_cls: sessionmaker[Session],
    dest_base_path: Path,
) -> list[CropTask]:
    with session_cls() as session:
        video = load_video(session, video_path, version)
        dest_path = dest_base_path / version / video.camera.name / video_path.name
        frame_features = sampler.sample(video.video_id, session)
        crop_tasks = [
            CropTask(
                feature.frame_nr, destination_path(dest_path, feature), BoundingBox.from_tracking_frame_feature(feature)
            )
            for feature in frame_features
        ]
    return crop_tasks


def crop_from_video(video_path: Path, crop_tasks: list[CropTask]) -> None:
    crop_queue = deque(sorted(crop_tasks))
    with video_reader(video_path) as video_feed:
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


def crop(
    video_path: Path,
    version: str,
    sampler: Sampler,
    session_cls: sessionmaker[Session],
    dest_base_path: Path,
) -> None:
    crop_tasks = create_crop_tasks(video_path, version, sampler, session_cls, dest_base_path)

    if not crop_tasks:
        log.warning(f"No frames to crop for video: {video_path}")
        return

    crop_from_video(video_path, crop_tasks)


# TODO(memben): cleanup
if __name__ == "__main__":
    import shutil

    from sqlalchemy import create_engine
    from tqdm import tqdm

    from gorillatracker.ssl_pipeline.queries import (
        confidence_filter,
        feature_type_filter,
        min_count_filter,
        video_filter,
    )

    # engine = create_engine("postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres")
    engine = create_engine("sqlite:///test.db")

    def sampling_strategy(video_id: int, min_n_images_per_tracking: int) -> Select[tuple[TrackingFrameFeature]]:
        query = video_filter(video_id)
        query = min_count_filter(query, min_n_images_per_tracking)
        query = feature_type_filter(query, [GorillaDataset.FACE_90])
        query = confidence_filter(query, 0.7)
        query = min_count_filter(query, 10)
        return query

    shutil.rmtree("cropped_images")

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-09"

    with session_cls() as session:
        videos = session.execute(select(Video)).scalars().all()
        video_paths = [Path(video.path) for video in videos]

    for video_path in tqdm(video_paths):
        query = partial(sampling_strategy, min_n_images_per_tracking=10)
        crop(
            video_path,
            version,
            RandomSampler(query_builder=query, seed=42, n_samples=10),
            session_cls,
            Path("cropped_images"),
        )
