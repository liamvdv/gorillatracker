"""
Contains adapter classes for different datasets.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.correlators import Correlator, one_to_one_correlator
from gorillatracker.ssl_pipeline.models import Base, Camera, Video
from gorillatracker.ssl_pipeline.video_tracker import VideoMetadata

log = logging.getLogger(__name__)


class SSLDataset(ABC):
    def __init__(self, db_uri: str) -> None:
        engine = create_engine(db_uri, echo=True)
        self._engine = engine

    def unprocessed_videos(self) -> list[Path]:
        """Returns a list of videos that have not been processed, this function is not Idempotent."""
        with Session(self._engine) as session:
            stmt = select(Video.filename)
            videos = session.scalars(stmt).all()
        log.info(f"Found {len(videos)} (processed) videos in the database")
        return [video for video in self.videos if video.name not in videos]

    def feature_models(self) -> list[tuple[Path, dict[str, Any], Correlator, str]]:
        """Returns a list of feature models to use for adding features of interest (e.g. face detector)"""
        return []

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
    def tracker_config(self) -> Path:
        pass

    @property
    @abstractmethod
    def yolo_kwargs(self) -> dict[str, Any]:
        # full list of kwargs: https://docs.ultralytics.com/modes/predict/#inference-arguments
        # reduce compute time with vid_stride (sample every nth frame), half (half size)
        # NOTE(memben): YOLOv8s video streaming has an internal off by one https://github.com/ultralytics/ultralytics/issues/8976 error, we fix it internally
        pass

    @property
    @abstractmethod
    def engine(self) -> Engine:
        pass

    @abstractmethod
    def setup_database(self) -> None:
        """Creates and populates the database. Population is dataset specific but should include cameras setup."""
        pass


class GorillaDataset(SSLDataset):
    _yolo_base_kwargs = {
        "half": True,  # We found no difference in accuracy to False
        "vid_stride": 5,
        "verbose": False,
    }
    
    DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"
    
    def __init__(self, db_uri: str = DB_URI) -> None:
        super().__init__(db_uri)

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

    def feature_models(self) -> list[tuple[Path, dict[str, Any], Correlator, str]]:
        return [
            (Path("models/yolov8n_gorilla_face_45.pt"), self._yolo_base_kwargs, one_to_one_correlator, "face_45"),
            (Path("models/yolov8n_gorilla_face_90.pt"), self._yolo_base_kwargs, one_to_one_correlator, "face_90"),
        ]

    @property
    def videos(self) -> list[Path]:
        return list(Path("video_data").glob("*.mp4"))

    @property
    def body_model(self) -> Path:
        return Path("models/yolov8n_gorilla_body.pt")

    @property
    def metadata_extractor(self) -> Callable[[Path], VideoMetadata]:
        return GorillaDataset.get_video_metadata

    @property
    def tracker_config(self) -> Path:
        return Path("cfgs/tracker/botsort.yaml")

    @property
    def yolo_kwargs(self) -> dict[str, Any]:
        # NOTE(memben): YOLOv8s video streaming has an internal off by one https://github.com/ultralytics/ultralytics/issues/8976 error, we fix it internally
        return {
            **self._yolo_base_kwargs,
            "iou": 0.2,
            "conf": 0.7,
        }

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
