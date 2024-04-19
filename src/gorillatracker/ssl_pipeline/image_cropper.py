from __future__ import annotations

import logging
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterator

import cv2
from sqlalchemy import Engine, Select, select, update
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.helpers import BoundingBox, crop_frame, video_reader
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
    tracking_frame_feature_id: int


def destination_path(base_path: Path, feature: TrackingFrameFeature) -> Path:
    return Path(base_path, str(feature.tracking.tracking_id), f"{feature.frame_nr}.png")


def create_crop_tasks(
    video_path: Path,
    version: str,
    session_cls: sessionmaker[Session],
    sampler: Sampler,
    dest_base_path: Path,
) -> list[CropTask]:
    with session_cls() as session:
        video = load_video(session, video_path, version)
        dest_path = dest_base_path / version / video.camera.name / video_path.name
        frame_features = sampler.sample(video.video_id, session)
        crop_tasks = [
            CropTask(
                feature.frame_nr,
                destination_path(dest_path, feature),
                BoundingBox.from_tracking_frame_feature(feature),
                feature.tracking_frame_feature_id,
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
                cropped_frame = crop_frame(video_frame.frame, crop_task.bounding_box)
                crop_task.dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_task.dest), cropped_frame)
        assert not crop_queue, "Not all crop tasks were completed"


def update_cache_paths(crop_tasks: list[CropTask], session_cls: sessionmaker[Session]) -> None:
    with session_cls() as session:
        set_where_statements = [
            {
                "tracking_frame_feature_id": crop_task.tracking_frame_feature_id,
                "cache_path": str(crop_task.dest),
            }
            for crop_task in crop_tasks
        ]

        session.execute(update(TrackingFrameFeature), set_where_statements)
        session.commit()


def crop(
    video_path: Path, version: str, session_cls: sessionmaker[Session], sampler: Sampler, dest_base_path: Path
) -> None:
    crop_tasks = create_crop_tasks(video_path, version, session_cls, sampler, dest_base_path)

    if not crop_tasks:
        log.warning(f"No frames to crop for video: {video_path}")
        return

    crop_from_video(video_path, crop_tasks)
    update_cache_paths(crop_tasks, session_cls)


_version = None
_session_cls = None
_sampler = None


def _init_cropper(engine: Engine, version: str, sampler: Sampler) -> None:
    global _version, _session_cls, _sampler
    _version = version
    _sampler = sampler
    engine.dispose(close=False)
    _session_cls = sessionmaker(bind=engine)


def _multiprocess_crop(
    video_path: Path,
    dest_base_path: Path,
) -> None:
    global _version, _session_cls, _sampler
    assert _session_cls is not None, "Engine not initialized, call _init_cropper first"
    assert _version is not None, "Version not initialized, call _init_cropper instead"
    assert _sampler is not None, "Sampler not initialized, call _init_cropper instead"
    crop(video_path, _version, _session_cls, _sampler, dest_base_path)


def multiprocess_crop_from_video(
    video_paths: list[Path], version: str, engine: Engine, sampler: Sampler, dest_base_path: Path, max_workers: int
) -> None:
    with ProcessPoolExecutor(
        initializer=_init_cropper, initargs=(engine, version, sampler), max_workers=max_workers
    ) as executor:
        list(
            tqdm(
                executor.map(_multiprocess_crop, video_paths, [dest_base_path] * len(video_paths)),
                total=len(video_paths),
                desc="Cropping images from videos",
                unit="video",
            )
        )


if __name__ == "__main__":
    import shutil

    from sqlalchemy import create_engine

    from gorillatracker.ssl_pipeline.queries import associated_filter, video_filter

    # engine = create_engine("postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres")
    engine = create_engine("sqlite:///test.db")

    def sampling_strategy(video_id: int, min_n_images_per_tracking: int) -> Select[tuple[TrackingFrameFeature]]:
        query = video_filter(video_id)
        query = associated_filter(query)
        return query

    shutil.rmtree("cropped_images")
    Path("cropped_images").mkdir(parents=True, exist_ok=True)

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-09"  # TODO(memben)

    with session_cls() as session:
        videos = session.execute(select(Video)).scalars().all()
        video_paths = [Path(video.path) for video in videos]

    query = partial(sampling_strategy, min_n_images_per_tracking=10)
    sampler = Sampler(query_builder=query)

    multiprocess_crop_from_video(video_paths[:20], version, engine, sampler, Path("cropped_images"), max_workers=10)

    # print cache_paths of first 50 TrackingFrameFeature instances
    with session_cls() as session:
        tracking_frame_features = session.execute(select(TrackingFrameFeature)).scalars().all()
        for feature in tracking_frame_features[:50]:
            print(feature.cache_path)
