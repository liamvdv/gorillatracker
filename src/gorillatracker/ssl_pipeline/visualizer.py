import logging
import multiprocessing
import os
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import Optional, Sequence

import cv2
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.helpers import BoundingBox, groupby_frame, jenkins_hash, video_reader
from gorillatracker.ssl_pipeline.models import TaskType, TrackingFrameFeature, Video
from gorillatracker.ssl_pipeline.queries import get_next_task, transactional_task

log = logging.getLogger(__name__)


def id_to_color(track_id: Optional[int]) -> tuple[int, int, int]:
    """Convert a tracking ID to a color. (BGR)"""
    if track_id is None:
        return 0, 0, 255
    hash_value = jenkins_hash(track_id)
    h = (hash_value % 360) / 360.0
    s = max(0.7, (hash_value // 360) % 2)
    v = 0.9
    r, g, b = hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def render_on_frame(
    frame: cv2.typing.MatLike,
    frame_features: Sequence[TrackingFrameFeature],
) -> cv2.typing.MatLike:
    for frame_feature in frame_features:
        bbox = BoundingBox.from_tracking_frame_feature(frame_feature)
        cv2.rectangle(frame, bbox.top_left, bbox.bottom_right, id_to_color(frame_feature.tracking_id), 2)
        label_id = frame_feature.tracking_id or "UNRESOLVED"
        label = f"{label_id} ({frame_feature.feature_type}, {frame_feature.confidence:.2f})"
        cv2.putText(
            frame,
            label,
            (bbox.x_top_left, bbox.y_top_left - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            id_to_color(frame_feature.tracking_id),
            3,
        )
    return frame


def visualize_video(video: Video, dest: Path) -> None:
    assert video.path.exists()
    os.makedirs(dest.parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    tracked_frames = groupby_frame(video.tracking_frame_features)
    tracked_video = cv2.VideoWriter(str(dest), fourcc, video.output_fps, (video.width, video.height))
    with video_reader(video.path, frame_step=video.frame_step) as source_video:
        for frame in source_video:
            render_on_frame(frame.frame, tracked_frames[frame.frame_nr])
            tracked_video.write(frame.frame)
        tracked_video.release()


def visualize_worker(
    dest_base: Path,
    engine: Engine,
) -> None:
    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    engine.dispose(close=False)

    with Session(engine) as session:
        for task in get_next_task(session, TaskType.VISUALIZE):
            with transactional_task(session, task):
                video = task.video
                dest = dest_base / video.path.name
                visualize_video(video, dest)


def multiprocess_visualize(
    dest_base: Path,
    engine: Engine,
    process_count: int,
) -> None:
    processes: list[multiprocessing.Process] = []
    for _ in range(process_count):
        process = multiprocessing.Process(target=visualize_worker, args=(dest_base, engine))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    log.info("Visualizing completed")
