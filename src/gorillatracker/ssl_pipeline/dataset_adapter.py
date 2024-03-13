"""
Contains adapter classes for different datasets.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable

from pyparsing import C

from gorillatracker.ssl_pipeline.video_tracker import TrackerConfig, VideoMetadata


class SSLDatasetAdapter(ABC):
    @abstractmethod
    def get_body_model(self) -> Path:
        pass

    @abstractmethod
    def get_metadata_extractor(self) -> Callable[[Path], VideoMetadata]:
        pass

    @abstractmethod
    def get_tracker_config(self) -> TrackerConfig:
        pass


class GorillaDatasetAdapter(SSLDatasetAdapter):
    def get_body_model(self) -> Path:
        return Path("models/yolov8n_gorilla_body.pt")

    def get_metadata_extractor(self, video: Path) -> Callable[[Path], VideoMetadata]:
        return GorillaDatasetAdapter.get_video_metadata

    def get_tracker_config(self) -> TrackerConfig:
        return TrackerConfig(frame_stride=10, tracker_config=Path("cfgs/tracker/botsort.yaml"))

    @staticmethod
    def get_video_metadata(video: Path) -> VideoMetadata:
        camera_name = video.stem.split("_")[0]
        _, date_str, _ = video.stem.split("_")
        timestamps_path = "data/derived_data/timestamps.json"
        with open(timestamps_path, "r") as f:
            timestamps = json.load(f)
        date = datetime.strptime(date_str, "%Y%m%d")
        timestamp = timestamps[video.stem]
        daytime = datetime.strptime(timestamp, "%I:%M %p")
        date = datetime.combine(date, daytime.time())
        return VideoMetadata(camera_name, date)
    
