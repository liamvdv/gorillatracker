import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker

from gorillatracker.ssl_pipeline.helpers import BoundingBox, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import load_video_by_filename

log = logging.getLogger(__name__)


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
    sampling_query: Select[tuple[TrackingFrameFeature]],
    dest_base_path: Path,
) -> None:
    with session_cls() as session:
        video_tracking = load_video_by_filename(session, video)
        dest_path = dest_base_path / video_tracking.camera.name / video_tracking.filename
        frame_features = session.execute(sampling_query).scalars().all()
        crop_tasks = [
            CropTask(
                feature.frame_nr, destination_path(dest_path, feature), BoundingBox.from_tracking_frame_feature(feature)
            )
            for feature in frame_features
        ]
        print(len(crop_tasks))
    return

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
    import shutil

    from sqlalchemy import create_engine
    from tqdm import tqdm

    from gorillatracker.ssl_pipeline.queries import (
        feature_type_filter,
        min_count_filter,
        random_sampling_filter,
        video_filter,
    )

    engine = create_engine("sqlite:///test.db")

    def sampling_strategy(
        *, video: Path, min_n_images_per_tracking: int, crop_n_images_per_tracking: int, seed: int | None = None
    ) -> Select[tuple[TrackingFrameFeature]]:
        query = video_filter(video)
        # query = min_count_filter(query, min_n_images_per_tracking)
        query = feature_type_filter(query, ["body"])
        query = random_sampling_filter(query, crop_n_images_per_tracking, seed)
        print(query)
        return query

    # shutil.rmtree("cropped_images")

    session_cls = sessionmaker(bind=engine)

    with session_cls() as session:
        videos = session.execute(select(Video)).scalars().all()
        query = video_filter

    for video in tqdm(videos):
        video_path = Path("video_data", video.filename)
        crop_from_video(
            video_path,
            session_cls,
            sampling_strategy(video=video_path, min_n_images_per_tracking=200, crop_n_images_per_tracking=1),
            Path("cropped_images"),
        )
        break

    # from gorillatracker.ssl_pipeline.visualizer import visualize_video
    # p = Path("video_data/R033_20220403_392.mp4")
    # visualize_video(p, session_cls, Path("R033_20220403_392.mp4"))
