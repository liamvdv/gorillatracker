import logging
import math
from colorsys import hsv_to_rgb
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import Sequence, Tuple

import cv2
from attr import dataclass
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

import gorillatracker.ssl_pipeline.helpers as helpers
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedFrame:
    frame_nr: int
    frame_features: list[TrackingFrameFeature]


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    hash_value = helpers.jenkins_hash(track_id)
    h = (hash_value % 360) / 360.0
    s = max(0.7, (hash_value // 360) % 2)
    v = 0.9
    r, g, b = hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def render_frame(
    frame: cv2.typing.MatLike,
    frame_features: Sequence[TrackingFrameFeature],
    width: int,
    height: int,
    tracking_id_map: dict[int, int],
) -> None:
    for frame_feature in frame_features:
        bbox = helpers.BoundingBox.from_yolo(
            frame_feature.bbox_x_center,
            frame_feature.bbox_y_center,
            frame_feature.bbox_width,
            frame_feature.bbox_height,
            width,
            height,
        )
        cv2.rectangle(frame, bbox.top_left, bbox.bottom_right, id_to_color(frame_feature.tracking_id), 2)
        label = f"{tracking_id_map[frame_feature.tracking_id]} ({frame_feature.type})"
        cv2.putText(
            frame,
            label,
            (bbox.x_top_left, bbox.y_top_left - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            id_to_color(frame_feature.tracking_id),
            3,
        )


def get_tracked_frames(session: Session, video: Video) -> list[TrackedFrame]:
    tracked_frames: list[TrackedFrame] = []

    stmt = (
        select(TrackingFrameFeature)
        .join(Tracking)
        .where(Tracking.video_id == video.video_id)
        .order_by(TrackingFrameFeature.frame_nr)
    )
    frame_feature_query = session.scalars(stmt).all()

    if len(frame_feature_query) < 10:
        log.warning(f"Video {video.filename} has less than 10 frames with tracked features")

    frame_features_grouped = {
        frame_nr: list(features) for frame_nr, features in groupby(frame_feature_query, key=lambda x: x.frame_nr)
    }

    for frame_nr in range(0, video.frames, video.frame_step):
        frame_features = frame_features_grouped.get(frame_nr, [])
        tracked_frame = TrackedFrame(frame_nr=frame_nr, frame_features=frame_features)
        tracked_frames.append(tracked_frame)

    assert len(tracked_frames) == math.ceil(video.frames / video.frame_step)
    return tracked_frames


def visualize_video(video: Path, engine: Engine, dest: Path) -> None:
    assert video.exists()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    with Session(engine) as session:
        video_tracking = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        tracked_frames = get_tracked_frames(session, video_tracking)
        tracking_ids = set(feature.tracking_id for frame in tracked_frames for feature in frame.frame_features)
        tracking_id_map = {id: i + 1 for i, id in enumerate(tracking_ids)}
        # NOTE: video_tracking is the tracked version of source_video
        source_video = cv2.VideoCapture(str(video))
        tracked_video = cv2.VideoWriter(
            str(dest), fourcc, video_tracking.sampled_fps, (video_tracking.width, video_tracking.height)
        )
        for tracked_frame in tracked_frames:
            if tracked_frame.frame_nr > 1000:
                break
            source_video.set(cv2.CAP_PROP_POS_FRAMES, tracked_frame.frame_nr)
            success, frame = source_video.read()
            assert success
            render_frame(
                frame, tracked_frame.frame_features, video_tracking.width, video_tracking.height, tracking_id_map
            )
            tracked_video.write(frame)

        tracked_video.release()
        source_video.release()


_engine = None


def _init_visualizer(engine: Engine) -> None:
    global _engine
    _engine = engine
    _engine.dispose(
        close=False
    )  # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork


def _visualize_video_process(video: Path, dest_dir: Path) -> None:
    global _engine
    assert _engine is not None, "Engine not initialized, call _init_visualizer first"
    visualize_video(video, _engine, dest_dir / video.name)


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
