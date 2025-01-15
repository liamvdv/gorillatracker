from __future__ import annotations

import datetime as dt
import enum
import os
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import sqlalchemy.types as types
from sqlalchemy import CheckConstraint, Dialect, ForeignKey, Index, MetaData, String, UniqueConstraint, event
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy.orm.mapper import Mapper

# NOTE(memben): create schema manually in the database
SCHEMA = os.getenv("SCHEMA", "public")


# WARNING(memben): Changing the class may affect the database
# The values of the enum are stored in the database.
class VideoRelationshipType(enum.Enum):
    NEGATIVE = "negative"  # Implies that all Trackings in the left video are not in the right video and vice versa
    POSITIVE = "positive"  # Implies that one or more Trackings could be in both videos


# WARNING(memben): Changing the class may affect the database
# The values of the enum are stored in the database.
class TrackingRelationshipType(enum.Enum):
    NEGATIVE = "negative"  # Implies that the Trackings are not the same
    POSITIVE = "positive"  # Implies that the Trackings are the same (animal)


# WARNING(memben): Changing the class may affect the database
# The values of the enum are stored in the database.
class TaskStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# WARNING(memben): Changing the class may affect the database
# The values of the enum are stored in the database.
class TaskType(enum.Enum):
    TRACK = "track"
    PREDICT = "predict"
    CORRELATE = "correlate"
    VISUALIZE = "visualize"


T = TypeVar("T", bound=enum.Enum)


class ExtensibleEnum(types.TypeDecorator[T]):
    """Stores values as strings, converts them to and from enums."""

    impl = types.String(255)
    cache_ok = True

    def __init__(self, enum_cls: Type[T]) -> None:
        super().__init__()
        self.enum_cls = enum_cls

    def process_bind_param(self, value: Optional[T], dialect: Dialect) -> Optional[str]:
        if value is None:
            return None
        assert isinstance(value.value, str), f"{value} not serializable to string"
        return value.value

    def process_result_value(self, value: Optional[str], dialect: Dialect) -> Optional[T]:
        if value is None:
            return None
        return self.enum_cls(value)


class Base(DeclarativeBase):
    __abstract__ = True
    metadata = MetaData(schema=SCHEMA)


class Camera(Base):
    __tablename__ = "camera"

    camera_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    latitude: Mapped[Optional[float]]
    longitude: Mapped[Optional[float]]

    videos: Mapped[list[Video]] = relationship(back_populates="camera", cascade="all, delete-orphan")

    @validates("latitude")
    def validate_latitude(self, key: str, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError(f"{key} must be between -90 and 90, is {value}")
        return value

    @validates("longitude")
    def validate_longitude(self, key: str, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError(f"{key} must be between -180 and 180, is {value}")
        return value

    def __repr__(self) -> str:
        return f"camera(id={self.camera_id}, name={self.name}, latitude={self.latitude}, longitude={self.longitude})"


class Video(Base):
    __tablename__ = "video"

    video_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str]
    absolute_path: Mapped[str]
    camera_id: Mapped[int] = mapped_column(ForeignKey("camera.camera_id"))
    start_time: Mapped[Optional[dt.datetime]]
    width: Mapped[int]
    height: Mapped[int]
    fps: Mapped[float]  # of the original video
    target_output_fps: Mapped[int]  # of the tracked video
    frames: Mapped[int]

    camera: Mapped[Camera] = relationship(back_populates="videos")
    features: Mapped[list[VideoFeature]] = relationship(back_populates="video", cascade="all, delete-orphan")
    tasks: Mapped[list[Task]] = relationship(back_populates="video", cascade="all, delete-orphan")

    trackings: Mapped[list[Tracking]] = relationship(back_populates="video", cascade="all, delete-orphan")
    tracking_frame_features: Mapped[list[TrackingFrameFeature]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("fps > target_output_fps", name="fps_gt_target_output_fps"),
        UniqueConstraint("absolute_path", "version"),
    )

    @validates("version")
    def validate_version(self, key: str, value: str) -> str:
        try:
            dt.datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"{key} must be in the format 'YYYY-MM-DD', is {value}")
        return value

    @validates("absolute_path")
    def validate_path(self, key: str, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"{key} must be an absolute path, is {value}")
        if not (value.lower().endswith(".mp4") or value.lower().endswith(".avi")):
            raise ValueError(f"{key} must end with '.mp4', '.avi', '.MP4' or '.AVI', is {value}")
        return value

    @validates("width", "height", "fps", "target_output_fps", "frames")
    def validate_positive(self, key: str, value: int) -> int:
        if value <= 0:
            raise ValueError(f"{key} must be positive, is {value}")
        return value

    @property
    def output_fps(self) -> float:
        """The actual frames per second of the video after sampling."""
        return self.fps / self.frame_step

    @property
    def frame_step(self) -> int:
        """The number of frames to skip when sampling the video."""
        return int(self.fps / self.target_output_fps)

    @property
    def duration(self) -> dt.timedelta:
        return dt.timedelta(seconds=self.frames / self.fps)

    @property
    def path(self) -> Path:
        return Path(self.absolute_path)

    def __repr__(self) -> str:
        return f"""video(id={self.video_id}, version={self.version}, path={self.path}, 
                camera_id={self.camera_id}, start_time={self.start_time}, fps={self.fps}, frames={self.frames})"""


class VideoFeature(Base):
    __tablename__ = "video_feature"

    video_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))
    feature_type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    video: Mapped[Video] = relationship(back_populates="features")

    __table_args__ = (UniqueConstraint("video_id", "feature_type"),)

    def __repr__(self) -> str:
        return f"video_feature(id={self.video_feature_id}, video_id={self.video_id}, feature_type={self.feature_type}, value={self.value})"


class Tracking(Base):
    """Represent a continuous tracking of an animal in a video.

    A tracking is a sequence of frames in which an animal is tracked.
    The tracking is represented by a list of TrackingFrameFeatures,
    which are the features of the animal in each frame.

    There can be multiple trackings of the same animal.
    """

    __tablename__ = "tracking"

    tracking_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))

    video: Mapped[Video] = relationship(back_populates="trackings")
    features: Mapped[list[TrackingFeature]] = relationship(back_populates="tracking", cascade="all, delete-orphan")
    frame_features: Mapped[list[TrackingFrameFeature]] = relationship(
        back_populates="tracking", cascade="all, delete-orphan"
    )

    @property
    def tracking_duration(self) -> dt.timedelta:
        fps = self.video.fps
        start_frame = min(self.frame_features, key=lambda x: x.frame_nr).frame_nr
        end_frame = max(self.frame_features, key=lambda x: x.frame_nr).frame_nr
        return dt.timedelta(seconds=(end_frame - start_frame) / fps)

    def __repr__(self) -> str:
        return f"tracking(id={self.tracking_id}, video_id={self.video_id})"


class TrackingFeature(Base):
    __tablename__ = "tracking_feature"

    tracking_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tracking_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracking_id"))
    feature_type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="features")

    __table_args__ = (UniqueConstraint("tracking_id", "feature_type"),)

    def __repr__(self) -> str:
        return f"tracking_feature(id={self.tracking_feature_id}, tracking_id={self.tracking_id}, feature_type={self.feature_type}, value={self.value})"


class TrackingFrameFeature(Base):
    """Represent the detected bounding box of a tracking feature in a frame."""

    __tablename__ = "tracking_frame_feature"

    tracking_frame_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"), index=True)  # NOTE(memben): Denormalized
    tracking_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tracking.tracking_id"), nullable=True)
    frame_nr: Mapped[int]
    bbox_x_center_n: Mapped[float]
    bbox_y_center_n: Mapped[float]
    bbox_width_n: Mapped[float]
    bbox_height_n: Mapped[float]
    bbox_width: Mapped[int]
    bbox_height: Mapped[int]
    confidence: Mapped[float]
    feature_type: Mapped[str] = mapped_column(String(255))
    cached: Mapped[bool] = mapped_column(default=False)

    tracking: Mapped[Tracking] = relationship(back_populates="frame_features")
    video: Mapped[Video] = relationship(back_populates="tracking_frame_features")

    @validates("bbox_x_center_n", "bbox_y_center_n", "bbox_width_n", "bbox_height_n", "confidence")
    def validate_normalization(self, key: str, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1, is {value}")
        return value

    @validates("frame_nr")
    def validate_frame_nr(self, key: str, frame_nr: int) -> int:
        if frame_nr % (self.video.frame_step) != 0:
            raise ValueError(f"frame_nr must be a multiple of {self.video.frame_step}, is {frame_nr}")
        return frame_nr

    def cache_path(self, base_path: Path) -> Path:
        return Path(
            base_path,
            str(self.tracking_frame_feature_id % 2**8),
            str(self.tracking_frame_feature_id % 2**16),
            f"{self.tracking_frame_feature_id}.png",
        )

    __table_args__ = (
        Index("idx_tracking_id_frame_nr_feature_type", "tracking_id", "frame_nr", "feature_type", unique=True),
    )

    def __lt__(self, other: TrackingFrameFeature) -> bool:
        return self.tracking_frame_feature_id < other.tracking_frame_feature_id

    def __repr__(self) -> str:
        return f"""tracking_frame_feature(id={self.tracking_frame_feature_id}, video_id={self.video_id} tracking_id={self.tracking_id}, 
        frame_nr={self.frame_nr}, bbox_x_center_n={self.bbox_x_center_n}, bbox_y_center_n={self.bbox_y_center_n}, bbox_width_n={self.bbox_width_n}, bbox_height_n={self.bbox_height_n},
        bbox_width={self.bbox_width}, bbox_height={self.bbox_height}, confidence={self.confidence}, feature_type={self.feature_type}, cached={self.cached})"""


class VideoRelationship(Base):
    __tablename__ = "video_relationship"

    video_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))
    right_video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))
    edge: Mapped[VideoRelationshipType] = mapped_column(ExtensibleEnum(VideoRelationshipType))
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_video: Mapped[Video] = relationship(foreign_keys=[left_video_id])

    right_video: Mapped[Video] = relationship(foreign_keys=[right_video_id])
    __table_args__ = (
        CheckConstraint(
            "left_video_id < right_video_id", name="left_video_id_lt_right_video_id"
        ),  # for the unique constraint
        UniqueConstraint("left_video_id", "right_video_id", "reason", "created_by"),
    )

    def __repr__(self) -> str:
        return f"video_relationship(id={self.video_relationship_id}, left_video_id={self.left_video_id}, right_video_id={self.right_video_id}, edge={self.edge}, reason={self.reason}, created_by={self.created_by})"


class TrackingRelationship(Base):
    __tablename__ = "tracking_relationship"

    tracking_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_tracking_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracking_id"))
    right_tracking_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracking_id"))
    edge: Mapped[TrackingRelationshipType] = mapped_column(ExtensibleEnum(TrackingRelationshipType))
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_tracking: Mapped[Tracking] = relationship(foreign_keys=[left_tracking_id])
    right_tracking: Mapped[Tracking] = relationship(foreign_keys=[right_tracking_id])

    __table_args__ = (
        CheckConstraint("left_tracking_id < right_tracking_id", name="left_tracking_id_lt_right_tracking_id"),
        UniqueConstraint("left_tracking_id", "right_tracking_id", "reason", "created_by"),
    )

    def __repr__(self) -> str:
        return f"""tracking_relationship(id={self.tracking_relationship_id}, left_tracking_id={self.left_tracking_id}, right_tracking_id={self.right_tracking_id}, 
    edge={self.edge}, reason={self.reason}, created_by={self.created_by})"""


class Task(Base):
    __tablename__ = "task"

    task_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))
    task_type: Mapped[TaskType] = mapped_column(ExtensibleEnum(TaskType))
    task_subtype: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[TaskStatus] = mapped_column(ExtensibleEnum(TaskStatus), default=TaskStatus.PENDING)
    retries: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.now(dt.timezone.utc))

    video: Mapped[Video] = relationship(back_populates="tasks")
    task_key_values: Mapped[list[TaskKeyValue]] = relationship(back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("video_id", "task_type", "task_subtype"),)

    def get_key_value(self, key: str) -> str:
        for kv in self.task_key_values:
            if kv.key == key:
                return kv.value
        raise KeyError(f"Key {key} not found in task {self.task_id}")

    def __repr__(self) -> str:
        return f"task(id={self.task_id}, video_id={self.video_id}, task_type={self.task_type}, status={self.status}, last_modified={self.updated_at})"


@event.listens_for(Task, "before_update")
def task_before_update(mapper: Mapper[Any], connection: Connection, target: Task) -> None:
    target.updated_at = dt.datetime.now(dt.timezone.utc)


class TaskKeyValue(Base):
    __tablename__ = "task_key_value"

    task_key_value_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task.task_id"))
    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))
    task: Mapped[Task] = relationship(back_populates="task_key_values")

    __table_args__ = (UniqueConstraint("task_id", "key"),)

    def __repr__(self) -> str:
        return (
            f"task_key_value(id={self.task_key_value_id}, task_id={self.task_id}, key={self.key}, value={self.value})"
        )


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        camera = Camera(name="Test", latitude=0, longitude=0)
        video = Video(
            absolute_path="/absolute/path/to/test.mp4",
            version="2024-04-03",
            camera=camera,
            start_time=dt.datetime.now(dt.timezone.utc),
            width=1920,
            height=1080,
            fps=30,
            target_output_fps=5,
            frames=100,
        )
        tracking = Tracking(video=video)
        tracking_frame_feature = TrackingFrameFeature(
            tracking=tracking,
            video=video,
            frame_nr=0,
            bbox_x_center_n=0.5,
            bbox_y_center_n=0.5,
            bbox_width_n=0.5,
            bbox_height_n=0.5,
            bbox_width=960,
            bbox_height=540,
            confidence=0.5,
            feature_type="test",
        )
        task = Task(
            video=video,
            task_type=TaskType.PREDICT,
            status=TaskStatus.COMPLETED,
            updated_at=dt.datetime.now(dt.timezone.utc),
        )
        task_key_value = TaskKeyValue(task=task, key="test", value="test")
        session.add_all([camera, video, tracking, tracking_frame_feature, task])
        session.commit()
    Base.metadata.drop_all(engine)
