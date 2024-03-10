from __future__ import annotations

import enum
from datetime import datetime, timedelta

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class VideoRelationshipType(enum.Enum):
    NEGATIVE = -1  # Implies that all Trackings in the left video are not in the right video and vice versa
    POSITIVE = 1  # Implies that one or more Trackings could be in both videos


class TrackingRelationshipType(enum.Enum):
    NEGATIVE = -1  # Implies that the Trackings are not the same
    POSITIVE = 1  # Implies that the Trackings are the same (animal)


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "Camera"

    camera_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    latitude: Mapped[float]
    longitude: Mapped[float]

    videos: Mapped[list[Video]] = relationship(back_populates="camera", cascade="all, delete-orphan")

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
    fps: Mapped[int]
    frames: Mapped[int]
    grayscale: Mapped[bool]

    camera: Mapped[Camera] = relationship("Camera", back_populates="videos")
    video_features: Mapped[list[VideoFeature]] = relationship(back_populates="video", cascade="all, delete-orphan")
    trackings: Mapped[list[Tracking]] = relationship(back_populates="video", cascade="all, delete-orphan")

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.frames / self.fps)

    def __repr__(self) -> str:
        return f"Video(id={self.video_id}, filename={self.filename}, camera_id={self.camera_id}, start_time={self.start_time}, fps={self.fps}, frames={self.frames}, gray={self.grayscale})"


class VideoFeature(Base):
    __tablename__ = "VideoFeature"

    feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    video: Mapped[Video] = relationship(back_populates="video_features")

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
    tracking_features: Mapped[list[TrackingFeature]] = relationship(
        back_populates="tracking", cascade="all, delete-orphan"
    )
    tracking_frame_features: Mapped[list[TrackingFrameFeature]] = relationship(
        back_populates="tracking", cascade="all, delete-orphan"
    )

    @property
    def tracking_duration(self) -> timedelta:
        fps = self.video.fps
        start_frame = min(self.tracking_frame_features, key=lambda x: x.frame_nr).frame_nr
        end_frame = max(self.tracking_frame_features, key=lambda x: x.frame_nr).frame_nr
        return timedelta(seconds=(end_frame - start_frame) / fps)

    def __repr__(self) -> str:
        return f"Tracking(id={self.tracking_id}, video_id={self.video_id})"


class TrackingFeature(Base):
    __tablename__ = "TrackingFeature"

    tracking_feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tracking_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    type: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(String(255))

    tracking: Mapped[Tracking] = relationship(back_populates="tracking_features")

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

    tracking: Mapped["Tracking"] = relationship(back_populates="tracking_frame_features")

    __table_args__ = (UniqueConstraint("tracking_id", "frame_nr", "type"),)

    def __repr__(self) -> str:
        return f"TrackingFrameFeature(id={self.tracking_frame_feature_id}, tracking_id={self.tracking_id}, frame_nr={self.frame_nr}, bbox_x_center={self.bbox_x_center}, bbox_y_center={self.bbox_y_center}, bbox_width={self.bbox_width}, bbox_height={self.bbox_height}, confidence={self.confidence}, type={self.type})"


class VideoRelationship(Base):
    __tablename__ = "VideoRelationship"

    video_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    right_video_id: Mapped[int] = mapped_column(ForeignKey("Video.video_id"))
    edge: Mapped[VideoRelationshipType]
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_video: Mapped[Video] = relationship(foreign_keys=[left_video_id])
    right_video: Mapped[Video] = relationship(foreign_keys=[right_video_id])

    def __repr__(self) -> str:
        return f"VideoRelationship(id={self.video_relationship_id}, left_video_id={self.left_video_id}, right_video_id={self.right_video_id}, edge={self.edge}, reason={self.reason}, created_by={self.created_by})"


class TrackingRelationship(Base):
    __tablename__ = "TrackingRelationship"

    tracking_relationship_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    left_video_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    right_video_id: Mapped[int] = mapped_column(ForeignKey("Tracking.tracking_id"))
    edge: Mapped[TrackingRelationshipType]
    reason: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(String(255))

    left_tracking: Mapped[Tracking] = relationship(foreign_keys=[left_video_id])
    right_tracking: Mapped[Tracking] = relationship(foreign_keys=[right_video_id])

    def __repr__(self) -> str:
        return f"TrackingRelationship(id={self.tracking_relationship_id}, left_video_id={self.left_video_id}, right_video_id={self.right_video_id}, edge={self.edge}, reason={self.reason}, created_by={self.created_by})"


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.close()
    Base.metadata.drop_all(engine)
