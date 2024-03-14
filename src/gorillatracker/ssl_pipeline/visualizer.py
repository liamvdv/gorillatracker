import logging
from colorsys import hsv_to_rgb
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path
from shutil import copyfile
from typing import Sequence, Tuple

import cv2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

import gorillatracker.ssl_pipeline.helpers as helpers
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video

log = logging.getLogger(__name__)


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    hash_value: int = ((track_id >> 16) ^ track_id) * 0x45D9F3B
    hash_value = ((hash_value >> 16) ^ hash_value) * 0x45D9F3B
    hash_value = (hash_value >> 16) ^ hash_value & 0xFFFFFFFF
    h = (hash_value % 360) / 360.0
    s = max(0.7, (hash_value // 360) % 2)
    v = 0.9
    r, g, b = hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def get_tracking_frame_features(session: Session, video: Video) -> Sequence[TrackingFrameFeature]:
    stmt = (
        select(TrackingFrameFeature)
        .join(Tracking)
        .where(Tracking.video_id == video.video_id)
        .order_by(TrackingFrameFeature.frame_nr)
    )
    tracking_frame_features = session.scalars(stmt).all()
    return tracking_frame_features


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


def get_sampled_fps(tracked_video: Video, frames: list[tuple[int, list[TrackingFrameFeature]]]) -> int:
    frame_distances = [next_frame[0] - frame[0] for frame, next_frame in zip(frames, frames[1:])]
    inferred_fps = tracked_video.fps / min(frame_distances)  # there might be multiple frames without a tracking

    assert inferred_fps - int(inferred_fps) < 1e-6, "Visualizer only supports full (int) FPS"
    sampled_fps = int(inferred_fps)

    assert all(
        frame_distance % int(tracked_video.fps / sampled_fps) == 0 for frame_distance in frame_distances
    ), "Visualizer only supports tracked videos with constant FPS"
    return sampled_fps


def visualize_video(video: Path, engine: Engine, dest: Path) -> None:
    assert video.exists()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    with Session(engine) as session:
        tracked_video = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        tracking_frame_features = get_tracking_frame_features(session, tracked_video)

        tracking_ids = set(f.tracking_id for f in tracking_frame_features)
        tracking_id_map = {id: i + 1 for i, id in enumerate(tracking_ids)}
        tracked_frames = [
            (frame_nr, list(frame_features))
            for frame_nr, frame_features in groupby(tracking_frame_features, key=lambda x: x.frame_nr)
        ]  # NOTE(memben): There can be frames without any tracked features
        if len(tracked_frames) < 2:
            log.warning(
                f"Video {video.name} has less than 2 frames with tracked features, saving the original video..."
            )
            copyfile(video, dest.parent / ("WARNING_" + dest.name))

        sampled_fps = get_sampled_fps(tracked_video, tracked_frames)
        assert sampled_fps > 0
        # tracked_video refrences source_video
        source_video = cv2.VideoCapture(str(video))
        output_video = cv2.VideoWriter(str(dest), fourcc, sampled_fps, (tracked_video.width, tracked_video.height))
        output_total_frames = tracked_video.fps * tracked_video.frames
        for source_frame_idx, tracked_frame in zip(
            range(0, output_total_frames, int(tracked_video.fps / sampled_fps)), tracked_frames
        ):
            output_frame_idx, frame_features = tracked_frame
            assert source_frame_idx == output_frame_idx
            source_video.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            success, frame = source_video.read()
            assert success
            render_frame(frame, frame_features, tracked_video.width, tracked_video.height, tracking_id_map)
            output_video.write(frame)

        output_video.release()
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
