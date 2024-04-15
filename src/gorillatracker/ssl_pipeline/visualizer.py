import logging
import os
from colorsys import hsv_to_rgb
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import cv2
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from gorillatracker.ssl_pipeline.helpers import BoundingBox, groupby_frame, jenkins_hash, video_reader
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature
from gorillatracker.ssl_pipeline.queries import load_video, video_filter

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
    tracking_id_to_label_map: dict[int, int],
) -> cv2.typing.MatLike:
    for frame_feature in frame_features:
        bbox = BoundingBox.from_tracking_frame_feature(frame_feature)
        cv2.rectangle(frame, bbox.top_left, bbox.bottom_right, id_to_color(frame_feature.tracking_id), 2)
        label = (
            f"{tracking_id_to_label_map[frame_feature.tracking_id]} ({frame_feature.type})"
            if frame_feature.tracking_id is not None
            else f"UNRESOLVED ({frame_feature.type})"
        )
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


def visualize_video(video_path: Path, version: str, session_cls: sessionmaker[Session], dest: Path) -> None:
    assert video_path.exists()
    os.makedirs(dest.parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    with session_cls() as session:
        video_tracking = load_video(session, video_path, version)
        tracked_frames = groupby_frame(session.scalars(video_filter(video_tracking.video_id)).all())
        unique_tracking_ids = {
            feature.tracking_id
            for frame in tracked_frames.values()
            for feature in frame
            if feature.tracking_id is not None
        }
        tracking_id_to_label_map = {id: i + 1 for i, id in enumerate(unique_tracking_ids)}
        # NOTE: video_tracking is the tracked version of source_video
        tracked_video = cv2.VideoWriter(
            str(dest), fourcc, video_tracking.sampled_fps, (video_tracking.width, video_tracking.height)
        )
        with video_reader(video_path, frame_step=video_tracking.frame_step) as source_video:
            for frame in source_video:
                render_on_frame(frame.frame, tracked_frames[frame.frame_nr], tracking_id_to_label_map)
                tracked_video.write(frame.frame)
            tracked_video.release()


_version = None
_session_cls = None


def _init_visualizer(engine: Engine, version: str) -> None:
    global _version, _session_cls
    _version = version
    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    _session_cls = sessionmaker(bind=engine)


def _visualize_video_process(video_path: Path, dest_dir: Path) -> None:
    global _version, _session_cls
    assert _session_cls is not None, "Engine not initialized, call _init_visualizer first"
    assert _version is not None, "Version not initialized, call _init_visualizer instead"
    visualize_video(video_path, _version, _session_cls, dest_dir / video_path.name)


def multiprocess_visualize_video(video_paths: list[Path], version: str, engine: Engine, dest_dir: Path) -> None:
    with ProcessPoolExecutor(initializer=_init_visualizer, initargs=(engine, version)) as executor:
        list(
            tqdm(
                executor.map(_visualize_video_process, video_paths, [dest_dir] * len(video_paths)),
                total=len(video_paths),
                desc="Visualizing videos",
                unit="video",
            )
        )
