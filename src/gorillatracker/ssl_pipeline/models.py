from __future__ import annotations

import enum
from datetime import datetime, timedelta

from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates


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


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "Camera"

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
        return f"Camera(id={self.camera_id}, name={self.name}, latitude={self.latitude}, longitude={self.longitude})"


class Video(Base):
    __tablename__ = "Video"

    video_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("Camera.camera_id"))
    start_time: Mapped[datetime]
    width: Mapped[int]
    height: Mapped[int]
    fps: Mapped[int]  # of the original video
    sampled_fps: Mapped[int]  # of the tracked video
    frames: Mapped[int]

    camera: Mapped[Camera] = relationship(back_populates="videos")
    features: Mapped[list[VideoFeature]] = relationship(back_populates="video", cascade="all, delete-orphan")
    trackings: Mapped[list[Tracking]] = relationship(back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("fps % sampled_fps = 0", name="fps_mod_sampled_fps"),)

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
    def duration(self) -> timedelta:
        return timedelta(seconds=self.frames / self.fps)

    def __repr__(self) -> str:
        return f"Video(id={self.video_id}, filename={self.filename}, camera_id={self.camera_id}, start_time={self.start_time}, fps={self.fps}, frames={self.frames})"


class VideoFeature(Base):
    __tablename__ = "VideoFeature"

    feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    video: Mapped[Video] = relationship(back_populates="features")

    def __repr__(self) -> str:
        return f"VideoFeature(id={self.feature_id}, video_id={self.video_id}, type={self.type}, value={self.value})"


class Tracking(Base):
    """Represent a continuous tracking of an animal in a video.

    A tracking is a sequence of frames in which an animal is tracked.
    The tracking is represented by a list of TrackingFrameFeatures,
    which are the features of the animal in each frame.

    There can be multiple trackings of the same animal.
    """

    __tablename__ = "Tracking"

    tracking_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))

    video: Mapped[Video] = relationship(back_populates="trackings")
    features: Mapped[list[TrackingFeature]] = relationship(back_populates="tracking", cascade="all, delete-orphan")
    frame_features: Mapped[list[TrackingFrameFeature]] = relationship(
        back_populates="tracking", cascade="all, delete-orphan"
    )

    @property
    def tracking_duration(self) -> timedelta:
        fps = self.video.fps
        start_frame = min(self.frame_features, key=lambda x: x.frame_nr).frame_nr
        end_frame = max(self.frame_features, key=lambda x: x.frame_nr).frame_nr
        return timedelta(seconds=(end_frame - start_frame) / fps)

    def __repr__(self) -> str:
        return f"Tracking(id={self.tracking_id}, video_id={self.video_id})"


class TrackingFeature(Base):
    __tablename__ = "TrackingFeature"

    tracking_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tracking_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="features")

    def __repr__(self) -> str:
        return f"TrackingFeature(id={self.tracking_feature_id}, tracking_id={self.tracking_id}, type={self.type}, value={self.value})"


class TrackingFrameFeature(Base):
    """Represent the detected bounding box of a tracking feature in a frame."""

    __tablename__ = "TrackingFrameFeature"

    tracking_frame_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tracking_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    frame_nr: Mapped[int]
    bbox_x_center: Mapped[float]
    bbox_y_center: Mapped[float]
    bbox_width: Mapped[float]
    bbox_height: Mapped[float]
    confidence: Mapped[float]
    type: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="frame_features")

    __table_args__ = (UniqueConstraint("tracking_id", "frame_nr", "type"),)

    @validates("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height", "confidence")
    def validate_normalization(self, key: str, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1, is {value}")
        return value

    @validates("frame_nr")
    def validate_frame_nr(self, key: str, frame_nr: int) -> int:
        video = self.tracking.video
        if frame_nr % (video.frame_step) != 0:
            raise ValueError(f"frame_nr must be a multiple of {video.frame_step}, is {frame_nr}")
        return frame_nr

    def __repr__(self) -> str:
        return f"TrackingFrameFeature(id={self.tracking_frame_feature_id}, tracking_id={self.tracking_id}, frame_nr={self.frame_nr}, bbox_x_center={self.bbox_x_center}, bbox_y_center={self.bbox_y_center}, bbox_width={self.bbox_width}, bbox_height={self.bbox_height}, confidence={self.confidence}, type={self.type})"


class VideoRelationship(Base):
    __tablename__ = "VideoRelationship"

    video_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    right_video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    edge: Mapped[str] = mapped_column(String(255))  # VideoRelationshipType
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_video: Mapped[Video] = relationship(foreign_keys=[left_video_id])

    right_video: Mapped[Video] = relationship(foreign_keys=[right_video_id])
    __table_args__ = (
        CheckConstraint("left_video_id < right_video_id", name="left_video_id_lt_right_video_id"),
        UniqueConstraint("left_video_id", "right_video_id", "reason", "created_by"),
    )

    @property
    def relationship(self) -> VideoRelationshipType:
        return VideoRelationshipType(self.edge)

    @validates("edge")
    def validate_edge(self, key: str, edge: str) -> str:
        allowed_values = [e.name for e in VideoRelationshipType]
        if edge not in allowed_values:
            raise ValueError(f"{key} must be one of {allowed_values}, not '{edge}'")
        return edge

    def __repr__(self) -> str:
        return f"VideoRelationship(id={self.video_relationship_id}, left_video_id={self.left_video_id}, right_video_id={self.right_video_id}, edge={self.edge}, reason={self.reason}, created_by={self.created_by})"


class TrackingRelationship(Base):
    __tablename__ = "TrackingRelationship"

    tracking_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_tracking_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    right_tracking_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    edge: Mapped[str] = mapped_column(String(255))  # TrackingRelationshipType
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_tracking: Mapped[Tracking] = relationship(foreign_keys=[left_tracking_id])
    right_tracking: Mapped[Tracking] = relationship(foreign_keys=[right_tracking_id])

    __table_args__ = (
        CheckConstraint("left_tracking_id < right_tracking_id", name="left_tracking_id_lt_right_tracking_id"),
        UniqueConstraint("left_tracking_id", "right_tracking_id", "reason", "created_by"),
    )

    @property
    def relationship(self) -> TrackingRelationshipType:
        return TrackingRelationshipType(self.edge)

    @validates("edge")
    def validate_edge(self, key: str, edge: str) -> str:
        allowed_values = [e.name for e in TrackingRelationshipType]
        if edge not in allowed_values:
            raise ValueError(f"{key} must be one of {allowed_values}, not '{edge}'")
        return edge

    def __repr__(self) -> str:
        return f"TrackingRelationship(id={self.tracking_relationship_id}, left_tracking_id={self.left_tracking_id}, right_tracking_id={self.right_tracking_id}, edge={self.edge}, reason={self.reason}, created_by={self.created_by})"


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.close()
    Base.metadata.drop_all(engine)