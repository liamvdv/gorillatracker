import logging
from colorsys import hsv_to_rgb
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

import gorillatracker.ssl_pipeline.helpers as helpers
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature, Video

log = logging.getLogger(__name__)


def id_to_color(track_id: int) -> tuple[int, int, int]:
    hash_value = helpers.jenkins_hash(track_id)
    h = (hash_value % 360) / 360.0
    s = max(0.7, (hash_value // 360) % 2)
    v = 0.9
    r, g, b = hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def render_frame(
    frame: cv2.typing.MatLike,
    frame_features: Sequence[TrackingFrameFeature],
    tracking_id_to_label_map: dict[int, int],
) -> None:
    for frame_feature in frame_features:
        bbox = helpers.BoundingBox.from_tracking_frame_feature(frame_feature)
        cv2.rectangle(frame, bbox.top_left, bbox.bottom_right, id_to_color(frame_feature.tracking_id), 2)
        label = f"{tracking_id_to_label_map[frame_feature.tracking_id]} ({frame_feature.type})"
        cv2.putText(
            frame,
            label,
            (bbox.x_top_left, bbox.y_top_left - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            id_to_color(frame_feature.tracking_id),
            3,
        )


def visualize_video(video: Path, session_cls: sessionmaker[Session], dest: Path) -> None:
    assert video.exists()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    with session_cls() as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        tracked_frames = helpers.get_tracked_frames(session, video_tracking)
        unique_tracking_ids = set(feature.tracking_id for frame in tracked_frames for feature in frame.frame_features)
        tracking_id_to_label_map = {id: i + 1 for i, id in enumerate(unique_tracking_ids)}
        # NOTE: video_tracking is the tracked version of source_video
        tracked_video = cv2.VideoWriter(
            str(dest), fourcc, video_tracking.sampled_fps, (video_tracking.width, video_tracking.height)
        )
        source_video = helpers.video_generator(video, frame_step=video_tracking.frame_step)
        for tracked_frame, source_frame in zip(tracked_frames, source_video, strict=True):
            render_frame(
                source_frame,
                tracked_frame.frame_features,
                tracking_id_to_label_map,
            )
            tracked_video.write(source_frame)
        tracked_video.release()


_session_cls = None


def _init_visualizer(engine: Engine) -> None:
    global _session_cls
    engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    _session_cls = sessionmaker(bind=engine)


def _visualize_video_process(video: Path, dest_dir: Path) -> None:
    global _session_cls
    assert _session_cls is not None, "Engine not initialized, call _init_visualizer first"
    visualize_video(video, _session_cls, dest_dir / video.name)


def multiprocess_visualize_video(videos: list[Path], engine: Engine, dest_dir: Path) -> None:
    with ProcessPoolExecutor(initializer=_init_visualizer, initargs=(engine,)) as executor:
        list(
            tqdm(
                executor.map(_visualize_video_process, videos, [dest_dir] * len(videos)),
                total=len(videos),
                desc="Visualizing videos",
                unit="video",
            )
        )
