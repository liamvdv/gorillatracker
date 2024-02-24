from __future__ import annotations

import datetime
import enum

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class RelationshipType(enum.Enum):
    NEGATIVE = -1
    POSITIVE = 1


class AnimalType(enum.Enum):
    SILVERBACK = "silverback"
    INFANT = "infant"
    GORILLA = "gorilla"  # generic gorilla


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "camera"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    latitude: Mapped[float]
    longitude: Mapped[float]

    video_clips: Mapped[list[Camera]] = relationship(back_populates="camera")


class VideoClip(Base):
    __tablename__ = "video_clip"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("camera.id"))
    start_time: Mapped[datetime.datetime]
    fps: Mapped[int]
    frames: Mapped[int]
    gray: Mapped[bool]

    camera: Mapped[Camera] = relationship(back_populates="video_clips")
    tracked_animals: Mapped[list[TrackedAnimal]] = relationship(back_populates="video")


class VideoRelationship(Base):
    __tablename__ = "video_relationship"

    video_id_1: Mapped[int] = mapped_column(ForeignKey("video_clip.id"), primary_key=True)
    video_id_2: Mapped[int] = mapped_column(ForeignKey("video_clip.id"), primary_key=True)
    relationship_type: Mapped[RelationshipType]

    video_1: Mapped[VideoClip] = relationship(foreign_keys=[video_id_1])
    video_2: Mapped[VideoClip] = relationship(foreign_keys=[video_id_2])


class TrackedAnimal(Base):
    __tablename__ = "tracked_animal"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_clip.id"))
    animal_type: Mapped[AnimalType]

    video: Mapped[VideoClip] = relationship(back_populates="tracked_animals")
    trackings: Mapped[list[Tracking]] = relationship(back_populates="tracked_animal")


class BoundingBoxMixin:
    x_center: Mapped[float]
    y_center: Mapped[float]
    width: Mapped[float]
    height: Mapped[float]
    confidence: Mapped[float]


class Tracking(Base, BoundingBoxMixin):
    __tablename__ = "tracking"

    tracked_animal_id: Mapped[int] = mapped_column(ForeignKey("tracked_animal.id"), primary_key=True)
    frame_number: Mapped[int] = mapped_column(primary_key=True)

    tracked_animal: Mapped[TrackedAnimal] = relationship(back_populates="trackings")
    feature_tracking: Mapped[list[FeatureTracking]] = relationship(back_populates="tracking")


class Feature(Base):
    __tablename__ = "feature"

    id: Mapped[int] = mapped_column(primary_key=True)
    feature_name: Mapped[str] = mapped_column(String(30))


class FeatureTracking(Base, BoundingBoxMixin):
    __tablename__ = "feature_tracking"

    tracked_animal_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracked_animal_id"), primary_key=True)
    frame_number: Mapped[int] = mapped_column(ForeignKey("tracking.frame_number"), primary_key=True)

    tracking: Mapped[Tracking] = relationship(back_populates="feature_tracking")
