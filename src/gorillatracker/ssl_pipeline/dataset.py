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
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.feature_mapper import Correlator, one_to_one_correlator
from gorillatracker.ssl_pipeline.models import Base, Camera
from gorillatracker.ssl_pipeline.video_preprocessor import VideoMetadata

log = logging.getLogger(__name__)


class SSLDataset(ABC):
    def __init__(self, db_uri: str) -> None:
        engine = create_engine(db_uri)  # , echo=True)
        self._engine = engine

    def feature_models(self) -> list[tuple[Path, dict[str, Any], Correlator, str]]:
        """Returns a list of feature models to use for adding features of interest (e.g. face detector)"""
        return []

    @property
    @abstractmethod
    def video_paths(self) -> list[Path]:
        """The videos to track."""
        pass

    @property
    @abstractmethod
    def body_model_path(self) -> Path:
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
    FACE_90 = "face_90"  # angle of the face -90 to 90 degrees from the camera
    FACE_45 = "face_45"  # angle of the face -45 to 45 degrees from the camera

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

    def setup_cameras(self) -> None:
        df = pd.read_csv("data/ground_truth/cxl/misc/Kamaras_coorHPF.csv", sep=";", decimal=",")
        df["Name"] = df["Name"].str.rstrip("x")
        with Session(self._engine) as session:
            for _, row in df.iterrows():
                camera = Camera(name=row["Name"], latitude=row["lat"], longitude=row["long"])
                session.add(camera)
            session.commit()

    def feature_models(self) -> list[tuple[Path, dict[str, Any], Correlator, str]]:
        return [
            (Path("models/yolov8n_gorilla_face_45.pt"), self._yolo_base_kwargs, one_to_one_correlator, self.FACE_45),
            (Path("models/yolov8n_gorilla_face_90.pt"), self._yolo_base_kwargs, one_to_one_correlator, self.FACE_90),
        ]

    @property
    def video_paths(self) -> list[Path]:
        return list(Path("/workspaces/gorillatracker/video_data").glob("*.mp4"))

    @property
    def body_model_path(self) -> Path:
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
    def get_video_metadata(video_path: Path) -> VideoMetadata:
        camera_name = video_path.stem.split("_")[0]
        _, date_str, _ = video_path.stem.split("_")
        timestamps_path = "data/derived_data/timestamps.json"
        with open(timestamps_path, "r") as f:
            timestamps = json.load(f)
        date = datetime.strptime(date_str, "%Y%m%d")
        timestamp = timestamps[video_path.stem]
        daytime = datetime.strptime(timestamp, "%I:%M %p")
        date = datetime.combine(date, daytime.time())
        return VideoMetadata(camera_name, date)
