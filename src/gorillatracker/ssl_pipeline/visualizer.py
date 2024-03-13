from colorsys import hsv_to_rgb
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import tqdm
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

import gorillatracker.ssl_pipeline.helpers as helpers
from gorillatracker.ssl_pipeline.models import Tracking, TrackingFrameFeature, Video


def visualize_video(video: Path, engine: Engine, dest: Path) -> None:
    def id_to_color(track_id: int) -> tuple[int, int, int]:
        hash: int = ((track_id >> 16) ^ track_id) * 0x45D9F3B
        hash = ((hash >> 16) ^ hash) * 0x45D9F3B
        hash = (hash >> 16) ^ hash & 0xFFFFFFFF
        h = (hash % 360) / 360.0
        s = max(0.7, (hash // 360) % 2)
        v = 0.9
        r, g, b = hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    source_video = cv2.VideoCapture(str(video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    Session = sessionmaker(bind=engine)
    with Session() as session:
        tracked_video = session.execute(select(Video).where(Video.filename == str(video.name))).scalar_one()
        stmt = (
            select(TrackingFrameFeature)
            .join(Tracking)
            .where(Tracking.video_id == tracked_video.video_id)
            .order_by(TrackingFrameFeature.frame_nr)
        )
        tracking_frame_features = session.scalars(stmt).all()
        tracking_ids = set(f.tracking_id for f in tracking_frame_features)
        tracking_id_map = {id: i + 1 for i, id in enumerate(tracking_ids)}
        frames = groupby(tracking_frame_features, key=lambda x: x.frame_nr)
        frames = list((frame_nr, list(frame_features)) for frame_nr, frame_features in frames)
        sampled_fps = int(tracked_video.fps / (frames[1][0] - frames[0][0]))
        for frame, next_frame in zip(frames, frames[1:]):
            assert int(tracked_video.fps / (next_frame[0] - frame[0])) == sampled_fps, f"{frame[0]} - {next_frame[0]}"
        assert all(
            int(tracked_video.fps / (next_frame[0] - frame[0])) == sampled_fps
            for frame, next_frame in zip(frames, frames[1:])
        ), "Visualizer only supports videos with constant FPS"
        frames = groupby(tracking_frame_features, key=lambda x: x.frame_nr)
        output_video = cv2.VideoWriter(str(dest), fourcc, sampled_fps, (tracked_video.width, tracked_video.height))

        frame_nr = 0
        frames = (f for f in frames)
        next_frame, frame_features = next(frames)
        while source_video.isOpened():
            success, frame = source_video.read()
            # TODO(memben): remove
            if frame_nr > 1000:
                break
            if not success:
                break
            if frame_nr != next_frame:
                frame_nr += 1
                continue
            frame_nr += 1

            for frame_feature in frame_features:
                bbox = helpers.BoundingBox.from_yolo(
                    frame_feature.bbox_x_center,
                    frame_feature.bbox_y_center,
                    frame_feature.bbox_width,
                    frame_feature.bbox_height,
                    tracked_video.width,
                    tracked_video.height,
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

            output_video.write(frame)
            try:
                next_frame, frame_features = next(frames)
            except StopIteration:
                break
        output_video.release()
    source_video.release()
