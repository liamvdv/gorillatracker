import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from gorillatracker.ssl_pipeline.helpers import BoundingBox, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import load_video_by_filename, load_video_tracking_frame_features

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


if __name__ == "__main__":
    engine = create_engine("sqlite:///test.db")
    video = Path("video_data/R019_20221228_607.mp4")
    session_cls = sessionmaker(bind=engine)
    crop_from_video(video, session_cls, lambda x: x, Path("cropped"))
    from gorillatracker.ssl_pipeline.visualizer import visualize_video
    visualize_video(video, session_cls, Path("visualized.mp4"))
    
