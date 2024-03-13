"""
Contains adapter classes for different datasets.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.models import Base, Camera
from gorillatracker.ssl_pipeline.video_tracker import TrackerConfig, VideoMetadata


class SSLDatasetAdapter(ABC):
    def __init__(self, db_uri: str) -> None:
        engine = create_engine(db_uri)
        self._engine = engine

    @property
    @abstractmethod
    def videos(self) -> list[Path]:
        """The videos to track."""
        pass

    @property
    @abstractmethod
    def body_model(self) -> Path:
        """The body model (YOLOv8) to use for tracking."""
        pass

    @property
    @abstractmethod
    def metadata_extractor(self) -> Callable[[Path], VideoMetadata]:
        """Function to extract metadata from video."""
        pass

    @property
    @abstractmethod
    def tracker_config(self) -> TrackerConfig:
        pass

    @property
    @abstractmethod
    def engine(self) -> Engine:
        pass

    @abstractmethod
    def setup_database(self) -> None:
        """Creates and populates the database. Population is dataset specific but should include cameras setup."""
        pass


class GorillaDatasetAdapter(SSLDatasetAdapter):
    def setup_database(self) -> None:
        Base.metadata.create_all(self._engine)
        self._setup_cameras()

    def _setup_cameras(self) -> None:
        df = pd.read_csv("data/ground_truth/cxl/misc/Kamaras_coorHPF.csv", sep=";", decimal=",")
        df["Name"] = df["Name"].str.rstrip("x")
        with Session(self._engine) as session:
            for _, row in df.iterrows():
                camera = Camera(name=row["Name"], latitude=row["lat"], longitude=row["long"])
                session.add(camera)
            session.commit()

    @property
    def videos(self) -> list[Path]:
        return list(Path("video_data").glob("*.mp4"))

    @property
    def body_model(self) -> Path:
        return Path("models/yolov8n_gorilla_body.pt")

    @property
    def metadata_extractor(self) -> Callable[[Path], VideoMetadata]:
        return GorillaDatasetAdapter.get_video_metadata

    @property
    def tracker_config(self) -> TrackerConfig:
        return TrackerConfig(frame_stride=10, tracker_config=Path("cfgs/tracker/botsort.yaml"))

    @property
    def engine(self) -> Engine:
        return self._engine

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
