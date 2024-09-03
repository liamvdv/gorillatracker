from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.helpers import BoundingBox, crop_frame, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import load_preprocessed_videos, load_video, video_filter

log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class CropTask:
    frame_nr: int
    dest: Path
    bounding_box: BoundingBox = field(compare=False)
    tracking_frame_feature_id: int


def create_crop_tasks(
    video_path: Path,
    version: str,
    session_cls: sessionmaker[Session],
    dest_base_path: Path,
    dest_base_path_squared: Path,
) -> list[CropTask]:
    with session_cls() as session:
        video = load_video(session, video_path, version)
        # NOTE(memben): When multiple videos have the same name, their TFF will be in the same folder, which is fine.
        # If we want to avoid this, we can add the video_id to the path.
        dest_path = dest_base_path / version
        query = video_filter(video.video_id)
        frame_features = session.execute(query).scalars().all()
        crop_tasks: list[CropTask] = []
        for feature in frame_features:
            crop_tasks.append(
                CropTask(
                    feature.frame_nr,
                    feature.cache_path(dest_path),
                    BoundingBox.from_tracking_frame_feature(feature),
                    feature.tracking_frame_feature_id,
                )
            )
            # squared
            crop_tasks.append(
                CropTask(
                    feature.frame_nr,
                    feature.cache_path(dest_base_path_squared / version),
                    BoundingBox.from_tracking_frame_feature_squared(feature),
                    feature.tracking_frame_feature_id,
                )
            )
    return crop_tasks


def crop_from_video(video_path: Path, crop_tasks: list[CropTask]) -> list[CropTask]:
    crop_queue = deque(sorted(crop_tasks))
    failed: list[CropTask] = []
    with video_reader(video_path) as video_feed:
        for video_frame in video_feed:
            while crop_queue and video_frame.frame_nr == crop_queue[0].frame_nr:
                crop_task = crop_queue.popleft()
                cropped_frame = crop_frame(video_frame.frame, crop_task.bounding_box)
                crop_task.dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cv2.imwrite(str(crop_task.dest), cropped_frame)
                except cv2.error as e:
                    log.error(f"Error writing cropped frame: {crop_task.dest}")
                    log.error(e)
                    failed.append(crop_task)
        assert not crop_queue, "Not all crop tasks were completed"
    return failed


def update_cached_tff(crop_tasks: list[CropTask], session_cls: sessionmaker[Session], failed: list[CropTask]) -> None:
    faild_task = set(failed)
    with session_cls() as session:
        set_where_statements = [
            {
                "tracking_frame_feature_id": crop_task.tracking_frame_feature_id,
                "cached": True,
            }
            for crop_task in crop_tasks
            if crop_task not in faild_task
        ]

        session.execute(update(TrackingFrameFeature), set_where_statements)
        session.commit()


def crop(
    video_path: Path,
    version: str,
    session_cls: sessionmaker[Session],
    dest_base_path: Path,
    dest_base_path_squared: Path,
) -> None:
    crop_tasks = create_crop_tasks(video_path, version, session_cls, dest_base_path, dest_base_path_squared)

    if not crop_tasks:
        log.warning(f"No frames to crop for video: {video_path}")
        return

    try:
        failed = crop_from_video(video_path, crop_tasks)
        update_cached_tff(crop_tasks, session_cls, failed)
    except cv2.error as e:
        log.error(f"Error cropping video: {video_path}")
        log.error(e)


_version = None
_session_cls = None


def _init_cropper(engine: Engine, version: str) -> None:
    global _version, _session_cls
    _version = version
    engine.dispose(close=False)
    _session_cls = sessionmaker(bind=engine)


def _multiprocess_crop(
    video_path: Path,
    dest_base_path: Path,
    dest_base_path_squared: Path,
) -> None:
    global _version, _session_cls
    assert _session_cls is not None, "Engine not initialized, call _init_cropper first"
    assert _version is not None, "Version not initialized, call _init_cropper instead"
    crop(video_path, _version, _session_cls, dest_base_path, dest_base_path_squared)


def multiprocess_crop_from_video(
    video_paths: list[Path],
    version: str,
    engine: Engine,
    dest_base_path: Path,
    dest_base_path_squared: Path,
    max_workers: int,
) -> None:
    with ProcessPoolExecutor(
        initializer=_init_cropper, initargs=(engine, version), max_workers=max_workers
    ) as executor:
        list(
            tqdm(
                executor.map(
                    _multiprocess_crop,
                    video_paths,
                    [dest_base_path] * len(video_paths),
                    [dest_base_path_squared] * len(video_paths),
                ),
                total=len(video_paths),
                desc="Cropping images from videos",
                unit="video",
            )
        )


if __name__ == "__main__":
    from sqlalchemy import create_engine

    RESET_CACHED = False

    engine = create_engine(GorillaDatasetKISZ.DB_URI)
    abs_path = Path("/workspaces/gorillatracker/video_data/cropped-images")
    abs_path_squared = Path("/workspaces/gorillatracker/video_data/cropped-images-squared")

    session_cls = sessionmaker(bind=engine)
    version = "2024-04-18"  # TODO(memben)

    with session_cls() as session:
        videos = load_preprocessed_videos(session, version)
        video_paths = [video.path for video in videos]
        if RESET_CACHED:
            batch_size = 100000
            max_tff_stmt = session.execute(
                select(func.max(TrackingFrameFeature.tracking_frame_feature_id))
            ).scalar_one()
            for i in tqdm(range(0, max_tff_stmt, batch_size), desc="Updating TrackingFrameFeatures"):
                update_stmt = (
                    update(TrackingFrameFeature)
                    .where(
                        i <= TrackingFrameFeature.tracking_frame_feature_id,
                        TrackingFrameFeature.tracking_frame_feature_id < i + batch_size,
                    )
                    .values(cached=False)
                )
                session.execute(update_stmt)
                session.commit()

    multiprocess_crop_from_video(video_paths, version, engine, abs_path, abs_path_squared, max_workers=80)
