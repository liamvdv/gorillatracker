from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass
from typing import Literal, Union

import dill as pickle
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
    split_by: Literal["percentage", "camera", "time", "custom"] = field(default="percentage")
    save_path: str = field(default=".")

    train_split: int = field(default=80)
    val_split: int = field(default=10)
    test_split: int = field(default=10)

    train_starttime: dt.datetime = field(default=dt.datetime(2010, 1, 1))
    train_endtime: dt.datetime = field(default=dt.datetime(2030, 1, 1))

    val_starttime: dt.datetime = field(default=dt.datetime(2010, 1, 1))
    val_endtime: dt.datetime = field(default=dt.datetime(2030, 1, 1))

    test_starttime: dt.datetime = field(default=dt.datetime(2010, 1, 1))
    test_endtime: dt.datetime = field(default=dt.datetime(2030, 1, 1))

    # include None to include videos without timestamp
    hours: list[Union[int, None]] = field(default_factory=lambda: list(range(0, 24)) + [None])

    video_length: tuple[int, int] = field(default=(0, 1000000))  # min, max video length in seconds

    max_train_videos: int = field(default=1000000)
    max_val_videos: int = field(default=1000000)
    max_test_videos: int = field(default=1000000)

    _train_video_ids: list[int] = field(init=False, default_factory=lambda: [])
    _val_video_ids: list[int] = field(init=False, default_factory=lambda: [])
    _test_video_ids: list[int] = field(init=False, default_factory=lambda: [])

    def _set_name(self) -> None:
        if self.split_by in ["percentage", "camera", "time"]:
            self.name = f"{self.name}_{self.version}_{self.split_by}-{self.train_split}-{self.val_split}-{self.test_split}_split"
        else:
            self.name = f"{self.name}_{self.version}_{self.split_by}_split"

        current_time = dt.datetime.now().strftime("%Y%m%d_%H%M")
        self.name = f"{self.name}_{current_time}"

    def train_video_ids(self) -> list[int]:
        return self._train_video_ids

    def val_video_ids(self) -> list[int]:
        return self._val_video_ids

    def test_video_ids(self) -> list[int]:
        return self._test_video_ids

    def create_split(self) -> None:
        self._set_name()
        if self.split_by == "percentage":
            self.split_by_percentage()
        elif self.split_by == "time":
            self.split_by_time()
        elif self.split_by == "camera":
            self.split_by_cameras()
        elif self.split_by == "custom":
            self.split_custom()
        else:
            raise ValueError("Invalid split_by argument")

    def build_query(
        self, range: tuple[dt.datetime, dt.datetime], max_videos: int, allow_none_date: bool = False
    ) -> Select[tuple[Video]]:
        query = vq.get_video_query(self.version)
        query = vq.date_filter(query, range, allow_none_date)
        query = vq.hour_filter(query, self.hours)
        query = vq.video_length_filter(query, self.video_length[0], self.video_length[1])
        query = vq.video_count_filter(query, max_videos)
        return query

    def build_train_query(self) -> Select[tuple[Video]]:
        query = self.build_query((self.train_starttime, self.train_endtime), self.max_train_videos)
        return query

    def build_val_query(self) -> Select[tuple[Video]]:
        query = self.build_query((self.val_starttime, self.val_endtime), self.max_val_videos)
        return query

    def build_test_query(self) -> Select[tuple[Video]]:
        query = self.build_query((self.test_starttime, self.test_endtime), self.max_test_videos)
        return query

    def split_by_time(self) -> None:
        """Split the videos based on the time range."""
        engine = create_engine(self.db_uri)
        session = sessionmaker(bind=engine)
        query = self.build_query((dt.datetime.min, dt.datetime.max), sys.maxsize)
        query = vq.order_by_time(query)
        videos = vq.get_videos_from_query(query, session())
        train_end = int(len(videos) * self.train_split / 100)
        val_end = train_end + int(len(videos) * self.val_split / 100)
        self._train_video_ids = (videos[:train_end])[: self.max_train_videos]
        self._val_video_ids = (videos[train_end:val_end])[: self.max_val_videos]
        self._test_video_ids = (videos[val_end:])[: self.max_test_videos]

    def split_by_percentage(self) -> None:
        """Split the videos based on the percentage split."""
        assert (
            self.train_split + self.val_split + self.test_split == 100
        ), "The sum of the split percentages must be 100"
        engine = create_engine(self.db_uri)
        session = sessionmaker(bind=engine)
        query = self.build_query((dt.datetime.min, dt.datetime.max), sys.maxsize, True)
        videos = vq.get_videos_from_query(query, session())
        train_end = int(len(videos) * self.train_split / 100)
        val_end = train_end + int(len(videos) * self.val_split / 100)
        self._train_video_ids = (videos[:train_end])[: self.max_train_videos]
        self._val_video_ids = (videos[train_end:val_end])[: self.max_val_videos]
        self._test_video_ids = (videos[val_end:])[: self.max_test_videos]

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
        )[: self.max_train_videos]
        self._val_video_ids = vq.get_videos_from_query(
            vq.camera_id_filter(self.build_val_query(), val_cameras), session()
        )[: self.max_val_videos]
        self._test_video_ids = vq.get_videos_from_query(
            vq.camera_id_filter(self.build_test_query(), test_cameras), session()
        )[: self.max_test_videos]

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

    def save_to_pickle(self) -> None:
        """Save the class instance to a pickle file."""
        file_path = os.path.join(self.save_path, f"{self.name}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_pickle(cls, path: str) -> SplitArgs:
        """Load the class instance from a pickle file."""
        with open(path, "rb") as file:
            return pickle.load(file)


if __name__ == "__main__":
    DB_URI = os.environ.get("POSTGRESQL_URI")
    if DB_URI is None:
        raise ValueError("Please set the DB_URI environment variable in devcontainer.json")
    args = SplitArgs(
        db_uri=DB_URI,
        version="2024-04-18",
        save_path="/workspaces/gorillatracker/data/splits/SSL/",
        split_by="percentage",
        train_split=80,
        val_split=10,
        test_split=10,
        train_starttime=dt.datetime(2010, 1, 1),
        train_endtime=dt.datetime(2030, 1, 1),
        val_starttime=dt.datetime(2010, 1, 1),
        val_endtime=dt.datetime(2030, 1, 1),
        test_starttime=dt.datetime(2010, 1, 1),
        test_endtime=dt.datetime(2030, 1, 1),
        hours=list(range(0, 24)),  # only videos from certain hours of the day
        video_length=(0, 1000000),  # min, max video length in seconds
        max_train_videos=1000,  # max videos in train bucket
        max_val_videos=10,  # max videos in val bucket
        max_test_videos=10,  # max videos in test bucket
    )
    if False:
        args.create_split()
        args.save_to_pickle()
        print("Split created and saved")
    else:
        split_path = "/workspaces/gorillatracker/data/splits/SSL/SSL-dev10vid_2024-04-18_percentage-100-0-0_split_20240606_1452.pkl"
        args = SplitArgs.load_pickle(split_path)
    print(len(args.train_video_ids()), len(args.val_video_ids()), len(args.test_video_ids()))
