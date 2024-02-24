from __future__ import annotations

import datetime
import enum
from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class RelationshipType(enum.Enum):
    NEGATIVE = -1
    POSITIVE = 1


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "camera"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(30))
    latitude: Mapped[float]
    longitude: Mapped[float]

    video_clips: Mapped[Optional[list[VideoClip]]] = relationship(back_populates="camera")

    def __repr__(self) -> str:
        return f"Camera(id={self.id}, name={self.name}, latitude={self.latitude}, longitude={self.longitude})"


class VideoClip(Base):
    __tablename__ = "video_clip"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("camera.id"))
    start_time: Mapped[datetime.datetime]
    fps: Mapped[int]
    frames: Mapped[int]
    width = Mapped[int]
    height = Mapped[int]
    grayscale: Mapped[bool]

    @property
    def duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.frames / self.fps)

    camera: Mapped[Camera] = relationship(back_populates="video_clips")
    tracked_animals: Mapped[Optional[list[TrackedAnimal]]] = relationship(back_populates="video")

    def __repr__(self) -> str:
        return f"VideoClip(id={self.id}, camera_id={self.camera_id}, start_time={self.start_time}, fps={self.fps}, frames={self.frames}, gray={self.grayscale})"


class VideoRelationship(Base):
    __tablename__ = "video_relationship"

    video_id_1: Mapped[int] = mapped_column(ForeignKey("video_clip.id"), primary_key=True)
    video_id_2: Mapped[int] = mapped_column(ForeignKey("video_clip.id"), primary_key=True)
    relationship_type: Mapped[RelationshipType]

    video_1: Mapped[VideoClip] = relationship(foreign_keys=[video_id_1])
    video_2: Mapped[VideoClip] = relationship(foreign_keys=[video_id_2])

    def __repr__(self) -> str:
        return f"VideoRelationship(video_id_1={self.video_id_1}, video_id_2={self.video_id_2}, relationship_type={self.relationship_type})"


class Animal(Base):
    __tablename__ = "animal"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(30))

    tracked_animals: Mapped[Optional[list[TrackedAnimal]]] = relationship(back_populates="animal")


class TrackedAnimal(Base):
    __tablename__ = "tracked_animal"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_clip.id"))
    animal_type: Mapped[int] = mapped_column(ForeignKey("animal.id"))

    animal: Mapped[Animal] = relationship(back_populates="tracked_animals")
    video: Mapped[VideoClip] = relationship(back_populates="tracked_animals")
    trackings: Mapped[list[Tracking]] = relationship(back_populates="tracked_animal")

    @property
    def tracking_duration(self) -> datetime.timedelta:
        fps = self.video.fps
        start_frame = min(self.trackings, key=lambda x: x.frame_number).frame_number
        end_frame = max(self.trackings, key=lambda x: x.frame_number).frame_number
        return datetime.timedelta(seconds=(end_frame - start_frame) / fps)

    def __repr__(self) -> str:
        return f"TrackedAnimal(id={self.id}, video_id={self.video_id}, animal_type={self.animal_type})"


class BoundingBoxMixin:
    x_center: Mapped[float]
    y_center: Mapped[float]
    width: Mapped[float]
    height: Mapped[float]
    confidence: Mapped[float]

    def __repr__(self) -> str:
        return f"BoundingBox(x_center={self.x_center}, y_center={self.y_center}, width={self.width}, height={self.height}, confidence={self.confidence})"


class Tracking(Base, BoundingBoxMixin):
    __tablename__ = "tracking"

    tracked_animal_id: Mapped[int] = mapped_column(ForeignKey("tracked_animal.id"), primary_key=True)
    frame_number: Mapped[int] = mapped_column(primary_key=True)

    tracked_animal: Mapped[TrackedAnimal] = relationship(back_populates="trackings")

    # TODO(@liamvdv): HELP
    # feature_trackings: Mapped[Optional[list[FeatureTracking]]] = relationship(back_populates="tracking", primaryjoin="Tracking.tracked_animal_id == FeatureTracking.tracked_animal_id and Tracking.frame_number == FeatureTracking.frame_number", foreign_keys=[tracked_animal_id, frame_number])
    def __repr__(self) -> str:
        return f"Tracking(tracked_animal_id={self.tracked_animal_id}, frame_number={self.frame_number}), {super().__repr__()}"


class Feature(Base):
    __tablename__ = "feature"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    feature_name: Mapped[str] = mapped_column(String(30))

    def __repr__(self) -> str:
        return f"Feature(id={self.id}, feature_name={self.feature_name})"


class FeatureTracking(Base, BoundingBoxMixin):
    __tablename__ = "feature_tracking"

    tracked_animal_id: Mapped[int] = mapped_column(ForeignKey("tracking.tracked_animal_id"), primary_key=True)
    frame_number: Mapped[int] = mapped_column(ForeignKey("tracking.frame_number"), primary_key=True)
    feature_id: Mapped[int] = mapped_column(ForeignKey("feature.id"), primary_key=True)

    # tracking: Mapped[Tracking] = relationship(back_populates="feature_tracking")

    def __repr__(self) -> str:
        return f"FeatureTracking(tracked_animal_id={self.tracked_animal_id}, frame_number={self.frame_number}), {super().__repr__()}"


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.close()
    Base.metadata.drop_all(engine)
