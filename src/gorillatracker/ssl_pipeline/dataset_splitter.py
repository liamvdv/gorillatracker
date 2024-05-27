import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Literal, Union

from simple_parsing import field
from sqlalchemy import Select, create_engine
from sqlalchemy.orm import sessionmaker

import gorillatracker.ssl_pipeline.video_filter_queries as vq
from gorillatracker.ssl_pipeline.models import Video


@dataclass(kw_only=True)  # type: ignore
class SplitArgs:
    db_uri: str
    version: str
    name: str = field(default="SSL-Video-Split")
    split_by: Literal["percentage", "camera", "custom"] = field(default="percentage")
    save_path: str = field(default=".")
    save_pickles: bool = field(default=False)
    save_json: bool = field(default=False)

    train_split: int = field(default=80)
    val_split: int = field(default=10)
    test_split: int = field(default=10)

    # include None to include videos without timestamp
    train_years: list[Union[int, None]] = field(default_factory=lambda: list(range(2010, 2030)) + [None])
    val_years: list[Union[int, None]] = field(default_factory=lambda: list(range(2010, 2030)) + [None])
    test_years: list[Union[int, None]] = field(default_factory=lambda: list(range(2010, 2030)) + [None])

    # include None to include videos without timestamp
    train_months: list[Union[int, None]] = field(default_factory=lambda: list(range(1, 13)) + [None])
    val_months: list[Union[int, None]] = field(default_factory=lambda: list(range(1, 13)) + [None])
    test_months: list[Union[int, None]] = field(default_factory=lambda: list(range(1, 13)) + [None])

    # include None to include videos without timestamp
    hours: list[Union[int, None]] = field(default_factory=lambda: list(range(0, 24)) + [None])

    video_length: tuple[int, int] = field(default=(0, 1000000))  # min, max video length in seconds

    max_train_videos: int = field(default=1000000)
    max_val_videos: int = field(default=1000000)
    max_test_videos: int = field(default=1000000)

    _train_video_ids: list[int] = field(init=False, default=[])
    _val_video_ids: list[int] = field(init=False, default=[])
    _test_video_ids: list[int] = field(init=False, default=[])

    def __post_init__(self) -> None:
        if self.split_by in ["percentage", "camera"]:
            self.name = f"{self.name}_{self.version}_{self.split_by}-{self.train_split}-{self.val_split}-{self.test_split}_split"
        else:
            self.name = f"{self.name}_{self.version}_{self.split_by}_split"
        self.create_split()
        if self.save_pickles:
            self.save_to_pickles()
        if self.save_json:
            self.save_to_json()

    def create_split(self) -> None:
        if self.split_by == "percentage":
            self.split_by_percentage()
        elif self.split_by == "camera":
            self.split_by_cameras()
        elif self.split_by == "custom":
            self.split_custom()
        else:
            raise ValueError("Invalid split_by argument")

    def build_train_query(self) -> Select[tuple[Video]]:
        query = vq.get_video_query(self.version)
        query = vq.year_filter(query, self.train_years)
        query = vq.month_filter(query, self.train_months)
        query = vq.hour_filter(query, self.hours)
        query = vq.video_length_filter(query, self.video_length[0], self.video_length[1])
        query = vq.video_count_filter(query, self.max_train_videos)
        return query

    def build_val_query(self) -> Select[tuple[Video]]:
        query = vq.get_video_query(self.version)
        query = vq.year_filter(query, self.val_years)
        query = vq.month_filter(query, self.val_months)
        query = vq.hour_filter(query, self.hours)
        query = vq.video_length_filter(query, self.video_length[0], self.video_length[1])
        query = vq.video_count_filter(query, self.max_val_videos)
        return query

    def build_test_query(self) -> Select[tuple[Video]]:
        query = vq.get_video_query(self.version)
        query = vq.year_filter(query, self.test_years)
        query = vq.month_filter(query, self.test_months)
        query = vq.hour_filter(query, self.hours)
        query = vq.video_length_filter(query, self.video_length[0], self.video_length[1])
        query = vq.video_count_filter(query, self.max_test_videos)
        return query

    def split_by_percentage(self) -> None:
        """Split the videos based on the percentage split."""
        assert (
            self.train_split + self.val_split + self.test_split == 100
        ), "The sum of the split percentages must be 100"
        query = self.build_train_query()
        engine = create_engine(self.db_uri)
        session = sessionmaker(bind=engine)
        videos = vq.get_videos_from_query(query, session())
        train_end = int(len(videos) * self.train_split / 100)
        val_end = train_end + int(len(videos) * self.val_split / 100)
        self._train_video_ids = videos[:train_end]
        self._val_video_ids = videos[train_end:val_end]
        self._test_video_ids = videos[val_end:]

    def split_by_cameras(self) -> None:
        """Split the videos based on camera ids."""
        assert (
            self.train_split + self.val_split + self.test_split == 100
        ), "The sum of the split percentages must be 100"
        engine = create_engine(self.db_uri)
        session = sessionmaker(bind=engine)
        cameras = vq.get_camera_ids(session())
        train_cameras = cameras[: int(len(cameras) * self.train_split / 100)]
        val_cameras = cameras[len(train_cameras) : len(train_cameras) + int(len(cameras) * self.val_split / 100)]
        test_cameras = cameras[len(train_cameras) + len(val_cameras) :]
        self._train_video_ids = vq.get_videos_from_query(
            vq.camera_id_filter(self.build_train_query(), train_cameras), session()
        )
        self._val_video_ids = vq.get_videos_from_query(
            vq.camera_id_filter(self.build_val_query(), val_cameras), session()
        )
        self._test_video_ids = vq.get_videos_from_query(
            vq.camera_id_filter(self.build_test_query(), test_cameras), session()
        )

    def split_custom(self) -> None:
        """Split the videos based on custom criteria."""
        engine = create_engine(self.db_uri)
        session = sessionmaker(bind=engine)
        self._train_video_ids = vq.get_videos_from_query(self.build_train_query(), session())
        val_query = self.build_val_query()
        val_query = vq.video_not_in(val_query, self._train_video_ids)
        self._val_video_ids = vq.get_videos_from_query(val_query, session())
        test_query = self.build_test_query()
        test_query = vq.video_not_in(test_query, self._train_video_ids + self._val_video_ids)
        self._test_video_ids = vq.get_videos_from_query(test_query, session())
        assert len(self._train_video_ids) != 0 and len(self._val_video_ids) != 0, "train and val cannot be empty"

    def config_to_json(self) -> dict[str, Any]:
        """Return the configuration as a dictionary."""
        return {
            "name": self.name,
            "split_by": self.split_by,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "train_years": self.train_years,
            "val_years": self.val_years,
            "test_years": self.test_years,
            "train_months": self.train_months,
            "val_months": self.val_months,
            "test_months": self.test_months,
            "hours": self.hours,
            "video_length": self.video_length,
            "max_train_videos": self.max_train_videos,
            "max_val_videos": self.max_val_videos,
            "max_test_videos": self.max_test_videos,
        }

    def save_to_json(self) -> None:
        """Write the split to a json file."""
        config = self.config_to_json()
        file_path = os.path.join(self.save_path, f"{self.name}.json")
        with open(file_path, "w") as f:
            json.dump(
                {
                    "config": config,
                    "train": self._train_video_ids,
                    "val": self._val_video_ids,
                    "test": self._test_video_ids,
                },
                f,
                indent=4,
            )

    def save_to_pickles(self) -> None:
        """Save the class instance to a pickle file."""
        file_path = os.path.join(self.save_path, f"{self.name}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(self, file)


if __name__ == "__main__":
    args = SplitArgs(
        db_uri="db_uri_here",
        version="2024-04-18",
        save_path="/workspaces/gorillatracker/data/splits/SSL/",
        split_by="percentage",
        save_pickles=True,
        save_json=False,
        train_split=80,
        val_split=10,
        test_split=10,
        train_years=list(range(2010, 2030)),  # only videos from certain years
        val_years=list(range(2010, 2030)),  # only videos from certain years
        test_years=list(range(2010, 2030)),  # only videos from certain years
        train_months=list(range(0, 13)),  # only videos from certain months of the year
        val_months=list(range(0, 13)),  # only videos from certain months of the year
        test_months=list(range(0, 13)),  # only videos from certain months of the year
        hours=list(range(0, 24)),  # only videos from certain hours of the day
        video_length=(0, 1000000),  # min, max video length in seconds
        max_train_videos=1000000,  # max videos in train bucket
        max_val_videos=1000000,  # max videos in val bucket
        max_test_videos=1000000,  # max videos in test bucket
    )
