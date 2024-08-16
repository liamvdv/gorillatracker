"""
Contains adapter classes for different datasets.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from gorillatracker.ssl_pipeline.helpers import extract_meta_data_time
from gorillatracker.ssl_pipeline.models import Base, Task, TaskKeyValue, TaskType, Video, VideoFeature
from gorillatracker.ssl_pipeline.queries import get_or_create_camera
from gorillatracker.ssl_pipeline.video_preprocessor import VideoMetadata

log = logging.getLogger(__name__)


# TODO(memben): rename this, name not changed for GitHub PR diff for now
class SSLDataset(ABC):
    BODY = "body"
    DB_URI = "sqlite:///test.db"

    def __init__(self, db_uri: str = DB_URI) -> None:
        if db_uri.startswith("sqlite"):
            log.warning("Using SQLite database, only one worker supported.")
        engine = create_engine(db_uri)
        self._engine = engine
        Base.metadata.create_all(self._engine)

    @property
    @abstractmethod
    def features(self) -> list[str]:
        """Returns a list of features of interest (e.g. face detector), body model NOT included."""
        pass

    @property
    @abstractmethod
    def video_paths(self) -> list[Path]:
        """The videos to track."""
        pass

    @property
    @abstractmethod
    def tracker_config(self) -> Path:
        """Path to the tracker config for the body model."""
        pass

    @property
    def engine(self) -> Engine:
        return self._engine

    @abstractmethod
    def get_yolo_model_config(self, name: str) -> tuple[Path, dict[str, Any]]:
        """Returns the path to the YOLO model and its kwargs."""
        # full list of kwargs: https://docs.ultralytics.com/modes/predict/#inference-arguments
        # NOTE(memben): YOLOv8s video streaming has an internal off by one https://github.com/ultralytics/ultralytics/issues/8976 error, we fix it internally
        pass

    @abstractmethod
    def metadata_extractor(self, video_path: Path) -> VideoMetadata:
        """Function to extract metadata from video."""
        pass

    @abstractmethod
    def post_setup(self) -> None:
        """Post setup operations."""
        pass

    @classmethod
    @abstractmethod
    def video_insert_hook(cls, video: Video) -> None:
        """Hook to run when inserting a video. Used to add Task's and VideoFeatures."""
        pass

    def drop_database(self) -> None:
        if input("Are you sure you want to drop the database? (y/n): ") == "y":
            Base.metadata.drop_all(self._engine)


class GorillaDataset(SSLDataset):
    FACE_90 = "face_90"  # angle of the face -90 to 90 degrees from the camera
    FACE_45 = "face_45"  # angle of the face -45 to 45 degrees from the camera
    VIDEO_DIR = "/workspaces/gorillatracker/video_data"
    CAMERA_LOCATIONS_CSV = "data/ground_truth/cxl/misc/Kamaras_coorHPF.csv"

    _yolo_base_kwargs = {
        "half": True,  # We found no difference in accuracy to False
        "verbose": False,
    }

    _yolo_body_kwargs = {
        **_yolo_base_kwargs,
        "iou": 0.2,
        "conf": 0.7,
    }

    _yolo_face_45_kwargs = {
        **_yolo_base_kwargs,
        "iou": 0.1,
    }

    _yolo_face_90_kwargs = {
        **_yolo_base_kwargs,
        "iou": 0.1,
    }

    _model_config = {
        SSLDataset.BODY: (Path("models/yolov8n_gorilla_body.pt"), _yolo_body_kwargs),
        FACE_45: (Path("models/yolov8n_gorilla_face_45.pt"), _yolo_face_45_kwargs),
        FACE_90: (Path("models/yolov8n_gorilla_face_90.pt"), _yolo_face_90_kwargs),
    }

    @property
    def features(self) -> list[str]:
        return [self.FACE_45, self.FACE_90]

    @property
    def tracker_config(self) -> Path:
        return Path("cfgs/tracker/botsort.yaml")

    @property
    def video_paths(self) -> list[Path]:
        return self.get_video_paths(self.VIDEO_DIR)

    @classmethod
    @abstractmethod
    def get_social_group(cls, video: Video) -> Optional[str]:
        pass

    def get_yolo_model_config(self, name: str) -> tuple[Path, dict[str, Any]]:
        return self._model_config[name]

    def get_video_paths(self, video_dir: str) -> list[Path]:
        videos = []
        for dirpath, _, filenames in os.walk(video_dir):
            for file in filenames:
                if file.startswith("."):
                    continue
                if file.lower().endswith((".avi", ".mp4")):
                    videos.append(Path(os.path.join(dirpath, file)))
        return videos

    def setup_camera_locations(self) -> None:
        df = pd.read_csv(self.CAMERA_LOCATIONS_CSV, sep=";", decimal=",")
        df["Name"] = df["Name"].str.rstrip("x")
        with Session(self._engine) as session:
            for _, row in df.iterrows():
                camera = get_or_create_camera(session, row["Name"])
                camera.latitude = row["lat"]
                camera.longitude = row["long"]
            session.commit()

    def post_setup(self) -> None:
        self.setup_camera_locations()

    @classmethod
    def video_insert_hook(cls, video: Video) -> None:
        social_group = cls.get_social_group(video)
        if social_group:
            video.features.append(VideoFeature(feature_type="social_group", value=social_group))

        tasks_to_add = [
            Task(task_type=TaskType.TRACK, task_subtype=cls.BODY),
            Task(task_type=TaskType.PREDICT, task_subtype=cls.FACE_90),
            Task(task_type=TaskType.PREDICT, task_subtype=cls.FACE_45),
            Task(
                task_type=TaskType.CORRELATE,
                task_subtype=cls.FACE_45,
                task_key_values=[
                    TaskKeyValue(key="tracked_feature_type", value=cls.BODY),
                    TaskKeyValue(key="untracked_feature_type", value=cls.FACE_45),
                    TaskKeyValue(key="threshold", value="0.7"),
                ],
            ),
            Task(
                task_type=TaskType.CORRELATE,
                task_subtype=cls.FACE_90,
                task_key_values=[
                    TaskKeyValue(key="tracked_feature_type", value=cls.BODY),
                    TaskKeyValue(key="untracked_feature_type", value=cls.FACE_90),
                    TaskKeyValue(key="threshold", value="0.7"),
                ],
            ),
        ]

        video.tasks.extend(tasks_to_add)


class GorillaDatasetBerlin(GorillaDataset):
    DB_URI: str = os.environ.get("POSTGRESQL_URI") or "sqlite:///:memory:"
    VIDEO_DIR = "/workspaces/gorillatracker/video_data/zoo_berlin/raw_collection_cleaned"

    def __init__(self, db_uri: str = DB_URI) -> None:
        assert os.environ.get(
            "POSTGRESQL_URI"
        ), "POSTGRESQL_URI environment variable not set, remove assertion if desired"
        super().__init__(db_uri)

        self._yolo_body_kwargs = {
            **self._yolo_base_kwargs,
            "iou": 0.2,
            "conf": 0.3,
        }

    @property
    def tracker_config(self) -> Path:
        return Path("cfgs/tracker/botsort_berlin.yml")

    # NOTE(memben): there is only one social group in the Berlin dataset
    @classmethod
    def get_social_group(cls, video: Video) -> Optional[str]:
        return None

    def find_camera(self, path: Path) -> str:
        parts = path.parts
        # NOTE(memben): our data being inconsitent zoo1, zoo_1 etc.
        matches = [part for part in parts if re.match(r"zoo.*\d+", part)]
        if len(matches) == 1:
            return matches[0]
        else:
            return "Unknown"

    def metadata_extractor(self, video_path: Path) -> VideoMetadata:
        camera_name = self.find_camera(video_path)
        return VideoMetadata(camera_name, None)

    def setup_camera_locations(self) -> None:
        # NOTE(memben): Berlin dataset does not have camera locations
        return None


class GorillaDatasetKISZ(GorillaDataset):
    DB_URI: str = os.environ.get("POSTGRESQL_URI") or "sqlite:///:memory:"

    def __init__(self, db_uri: str = DB_URI) -> None:
        assert os.environ.get(
            "POSTGRESQL_URI"
        ), "POSTGRESQL_URI environment variable not set, remove assertion if desired"
        super().__init__(db_uri)

    @classmethod
    def get_social_group(cls, video: Video) -> Optional[str]:
        parent = video.path.parent
        if not re.match(r"^.*?_\d+\s[A-Z]{2}$", parent.name):
            return None
        group_id = parent.name.split(" ")[1]
        if group_id == "XX":  # XX is unknown group
            return None
        return group_id

    def metadata_extractor(self, video_path: Path) -> VideoMetadata:
        camera_name = video_path.stem.split("_")[0]
        try:
            date = extract_meta_data_time(video_path)
        except ValueError:
            date = None
        return VideoMetadata(camera_name, date)


class GorillaDatasetGPUServer2(GorillaDataset):
    DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"
    TIMESTAMPS = "data/derived_data/timestamps.json"
    SOCIAL_GROUPS = "data/ground_truth/cxl/misc/VideosGO_SPAC.csv"

    def __init__(self, db_uri: str = DB_URI) -> None:
        super().__init__(self.DB_URI)

    @classmethod
    def get_social_group(cls, video: Video) -> Optional[str]:
        df = pd.read_csv(cls.SOCIAL_GROUPS, sep=",")
        filename = video.path.name.replace(".mp4", ".MP4")  # csv has .MP4 instead of .mp4
        matching_rows = df[df["File"] == filename]
        if not len(matching_rows) == 1:
            log.info(f"Found {len(matching_rows)} social groups for video {filename}, skipping")
            return None
        social_group = matching_rows.iloc[0]["Group"]
        if re.match(r"Group_[A-Z]{2}$", social_group):
            return social_group.split("_")[1]
        return None

    def metadata_extractor(self, video_path: Path) -> VideoMetadata:
        with open(self.TIMESTAMPS, "r") as f:
            timestamps = json.load(f)
        camera_name = video_path.stem.split("_")[0]
        _, date_str, _ = video_path.stem.split("_")
        date = datetime.strptime(date_str, "%Y%m%d")
        timestamp = timestamps[video_path.stem]
        daytime = datetime.strptime(timestamp, "%I:%M %p")
        date = datetime.combine(date, daytime.time())
        return VideoMetadata(camera_name, date)
