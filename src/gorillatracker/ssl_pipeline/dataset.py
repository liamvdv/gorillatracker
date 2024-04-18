"""
Contains adapter classes for different datasets.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import easyocr
import pandas as pd
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.feature_mapper import Correlator, one_to_one_correlator
from gorillatracker.ssl_pipeline.helpers import BoundingBox, extract_meta_data_time, read_timestamp
from gorillatracker.ssl_pipeline.models import Base, Camera, Video, VideoFeature
from gorillatracker.ssl_pipeline.video_preprocessor import MetadataExtractor, MetadataExtractorSmall, VideoMetadata

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
    def metadata_extractor(self) -> MetadataExtractor:
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
    TIME_STAMP_BOX: BoundingBox = BoundingBox(
        x_center_n=0.672969,
        y_center_n=0.978102,
        width_n=0.134167,
        height_n=0.043796,
        confidence=1,
        image_width=1920,
        image_height=1080,
    )  # default where time stamp is located

    _yolo_base_kwargs = {
        "half": True,  # We found no difference in accuracy to False
        "verbose": False,
    }

    DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"
    VIDEO_DIR = "/workspaces/gorillatracker/video_data"

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

    def setup_social_groups(self, version: str) -> None:
        feature_type = "social_group"
        video_groups = GorillaDataset.get_video_groups(self.VIDEO_DIR)
        with Session(self._engine) as session:
            for video_name, group_id in video_groups:
                try:
                    video_id = session.execute(
                        select(Video.video_id).where((Video.version == version) & (Video.path.endswith(video_name)))
                    ).scalar_one()
                except NoResultFound:
                    continue
                except MultipleResultsFound as e:
                    log.error(f"Multiple videos found in DB for {video_name} with version: {version}")
                    raise e
                video_feature = VideoFeature(video_id=video_id, type=feature_type, value=group_id)
                session.add(video_feature)
                try:
                    session.commit()
                except IntegrityError:
                    log.error(
                        f"Failed to add social group {group_id} for video {video_name} due to entry with video_id:{video_id} type:{feature_type} already in DB"
                    )
                    session.rollback()

    def feature_models(self) -> list[tuple[Path, dict[str, Any], Correlator, str]]:
        return [
            (Path("models/yolov8n_gorilla_face_45.pt"), self._yolo_base_kwargs, one_to_one_correlator, self.FACE_45),
            (Path("models/yolov8n_gorilla_face_90.pt"), self._yolo_base_kwargs, one_to_one_correlator, self.FACE_90),
        ]

    @property
    def video_paths(self) -> list[Path]:
        return GorillaDataset.get_video_paths(self.VIDEO_DIR)

    @property
    def body_model_path(self) -> Path:
        return Path("models/yolov8n_gorilla_body.pt")

    @property
    def metadata_extractor(self) -> MetadataExtractor:
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
        date = extract_meta_data_time(video_path)
        if date is None:
            log.error(f"Could not extract date from video {video_path}")
            raise ValueError(f"Could not extract date from video {video_path}")
        return VideoMetadata(camera_name, date)

    @staticmethod
    def get_video_paths(video_dir: str) -> list[Path]:
        videos = []
        for dirpath, dirnames, filenames in os.walk(video_dir):
            for file in filenames:
                if not file.startswith(".") and (file.lower().endswith(".avi") or file.lower().endswith(".mp4")):
                    videos.append(Path(os.path.join(dirpath, file)))
        return videos

    @staticmethod
    def get_video_groups(video_dir: str) -> list[Tuple[str, str]]:
        video_group_list = []
        for dirpath, dirnames, filenames in os.walk(video_dir):
            for file in filenames:
                if not file.startswith(".") and (file.lower().endswith(".avi") or file.lower().endswith(".mp4")):
                    file_path = Path(file)
                    parent = file_path.parent
                    if not re.match(r"^.*?_\d+\s[A-Z]{2}$", parent.name):
                        continue
                    group_id = parent.name.split(" ")[1]
                    if group_id == "XX":  # XX is unknown group
                        continue
                    video_group_list.append((file_path.name, group_id))
        return video_group_list


class GorillaDatasetSmall(SSLDataset):
    FACE_90 = "face_90"  # angle of the face -90 to 90 degrees from the camera
    FACE_45 = "face_45"  # angle of the face -45 to 45 degrees from the camera
    TIME_STAMP_BOX: BoundingBox = BoundingBox(
        x_center_n=0.672969,
        y_center_n=0.978102,
        width_n=0.134167,
        height_n=0.043796,
        confidence=1,
        image_width=1920,
        image_height=1080,
    )  # default where time stamp is located

    _yolo_base_kwargs = {
        "half": True,  # We found no difference in accuracy to False
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
    def metadata_extractor(self) -> MetadataExtractorSmall:
        return GorillaDatasetSmall.get_video_metadata

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
    def get_video_metadata(video_path: Path, ocr_reader: Optional[easyocr.Reader] = None) -> VideoMetadata:
        camera_name = video_path.stem.split("_")[0]
        _, date_str, _ = video_path.stem.split("_")
        date = datetime.strptime(date_str, "%Y%m%d")
        daytime = read_timestamp(video_path, GorillaDatasetSmall.TIME_STAMP_BOX, ocr_reader=ocr_reader)
        date = datetime.combine(date, daytime)
        return VideoMetadata(camera_name, date)
