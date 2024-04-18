from __future__ import annotations

import datetime as dt
import enum
from typing import Generic, Optional, Type, TypeVar

import sqlalchemy.types as types
from sqlalchemy import CheckConstraint, Dialect, ForeignKey, String, UniqueConstraint, event
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy.orm.mapper import Mapper


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
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


T = TypeVar("T", bound=enum.Enum)


class ExtensibleEnum(types.TypeDecorator, Generic[T]):
    """Stores values as strings, converts them to and from enums."""

    impl = types.String
    cache_ok = True

    def __init__(self, enum_cls: Type[T]) -> None:
        super().__init__()
        self.enum_cls = enum_cls

    def process_bind_param(self, value: T, dialect: Dialect) -> str:
        assert isinstance(value.value, str), f"{value} not serializable to string"
        return value.value

    def process_result_value(self, value: str, dialect: Dialect) -> T:
        return self.enum_cls(value)


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "camera"

    camera_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    latitude: Mapped[float]
    longitude: Mapped[float]

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
    path: Mapped[str]  # absolute path to the video file
    camera_id: Mapped[int] = mapped_column(ForeignKey("camera.camera_id"))
    start_time: Mapped[dt.datetime]
    width: Mapped[int]
    height: Mapped[int]
    fps: Mapped[int]  # of the original video
    sampled_fps: Mapped[int]  # of the tracked video
    frames: Mapped[int]

    camera: Mapped[Camera] = relationship(back_populates="videos")
    features: Mapped[list[VideoFeature]] = relationship(back_populates="video", cascade="all, delete-orphan")
    tasks: Mapped[list[Task]] = relationship(back_populates="video", cascade="all, delete-orphan")

    trackings: Mapped[list[Tracking]] = relationship(back_populates="video", cascade="all, delete-orphan")
    tracking_frame_features: Mapped[list[TrackingFrameFeature]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("fps % sampled_fps = 0", name="fps_mod_sampled_fps"),
        UniqueConstraint("path", "version"),
    )

    @validates("version")
    def validate_version(self, key: str, value: str) -> str:
        try:
            dt.datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"{key} must be in the format 'YYYY-MM-DD', is {value}")
        return value

    @validates("path")
    def validate_path(self, key: str, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"{key} must be an absolute path, is {value}")
        if not value.endswith(".mp4"):
            raise ValueError(f"{key} must end with '.mp4', is {value}")
        return value

    @validates("width", "height", "fps", "sampled_fps", "frames")
    def validate_positive(self, key: str, value: int) -> int:
        if value <= 0:
            raise ValueError(f"{key} must be positive, is {value}")
        return value

    @property
    def frame_step(self) -> int:
        """The number of frames to skip when sampling the video."""
        return self.fps // self.sampled_fps

    @property
    def duration(self) -> dt.timedelta:
        return dt.timedelta(seconds=self.frames / self.fps)

    def __hash__(self) -> int:
        return self.video_id

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

    def __hash__(self) -> int:
        return self.tracking_id

    def __repr__(self) -> str:
        return f"tracking(id={self.tracking_id}, video_id={self.video_id})"


class TrackingFeature(Base):
    __tablename__ = "tracking_feature"

    tracking_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tracking_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracking_id"))
    feature_type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="features")

    def __repr__(self) -> str:
        return f"tracking_feature(id={self.tracking_feature_id}, tracking_id={self.tracking_id}, feature_type={self.feature_type}, value={self.value})"


class TrackingFrameFeature(Base):
    """Represent the detected bounding box of a tracking feature in a frame."""

    __tablename__ = "tracking_frame_feature"

    tracking_frame_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))  # NOTE(memben): Denormalized
    tracking_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tracking.tracking_id"), nullable=True)
    frame_nr: Mapped[int]
    bbox_x_center: Mapped[float]
    bbox_y_center: Mapped[float]
    bbox_width: Mapped[float]
    bbox_height: Mapped[float]
    confidence: Mapped[float]
    feature_type: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="frame_features")
    video: Mapped[Video] = relationship(back_populates="tracking_frame_features")

    __table_args__ = (UniqueConstraint("tracking_id", "frame_nr", "feature_type"),)

    @validates("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height", "confidence")
    def validate_normalization(self, key: str, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1, is {value}")
        return value

    @validates("frame_nr")
    def validate_frame_nr(self, key: str, frame_nr: int) -> int:
        if frame_nr % (self.video.frame_step) != 0:
            raise ValueError(f"frame_nr must be a multiple of {self.video.frame_step}, is {frame_nr}")
        return frame_nr

    def __hash__(self) -> int:
        return self.tracking_frame_feature_id

    def __lt__(self, other: TrackingFrameFeature) -> bool:
        return self.tracking_frame_feature_id < other.tracking_frame_feature_id

    def __repr__(self) -> str:
        return f"""tracking_frame_feature(id={self.tracking_frame_feature_id}, video_id={self.video_id} tracking_id={self.tracking_id}, 
        frame_nr={self.frame_nr}, bbox_x_center={self.bbox_x_center}, bbox_y_center={self.bbox_y_center}, bbox_width={self.bbox_width}, 
        bbox_height={self.bbox_height}, confidence={self.confidence}, feature_type={self.feature_type})"""


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
    task_type: Mapped[str] = mapped_column(String(255))
    status: Mapped[TaskStatus] = mapped_column(ExtensibleEnum(TaskStatus), default=TaskStatus.QUEUED)
    retries: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.now(dt.timezone.utc))

    video: Mapped[Video] = relationship(back_populates="tasks")

    __table_args__ = (UniqueConstraint("video_id", "task_type"),)

    def __repr__(self) -> str:
        return f"task(id={self.task_id}, video_id={self.video_id}, task_type={self.task_type}, status={self.status}, last_modified={self.updated_at})"


@event.listens_for(Task, "before_update")
def task_before_update(mapper: Mapper, connection: Connection, target: Task):
    target.updated_at = dt.datetime.now(dt.timezone.utc)


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        camera = Camera(name="Test", latitude=0, longitude=0)
        video = Video(
            path="/absolute/path/to/test.mp4",
            version="2024-04-03",
            camera=camera,
            start_time=dt.datetime.now(dt.timezone.utc),
            width=1920,
            height=1080,
            fps=30,
            sampled_fps=5,
            frames=100,
        )
        tracking = Tracking(video=video)
        tracking_frame_feature = TrackingFrameFeature(
            tracking=tracking,
            video=video,
            frame_nr=0,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.5,
            bbox_height=0.5,
            confidence=0.5,
            feature_type="test",
        )
        task = Task(
            video=video, task_type="test", status=TaskStatus.COMPLETED, updated_at=dt.datetime.now(dt.timezone.utc)
        )
        session.add_all([camera, video, tracking, tracking_frame_feature, task])
        session.commit()
    Base.metadata.drop_all(engine)
