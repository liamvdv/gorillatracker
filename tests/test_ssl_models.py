"""
from __future__ import annotations

import datetime
import enum
from typing import Optional

from sqlalchemy import ForeignKey, Integer, String
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

    tracked_animal_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_animal.id"), primary_key=True
    )
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
    camera = Camera(name="camera1", latitude=0.0, longitude=0.0)
    session.add(camera)
    print(camera)
    # for s in session.query(Camera).all():
    #     print(s)
    session.close()
    Base.metadata.drop_all(engine)

"""

import datetime
from typing import Generator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from gorillatracker.ssl_pipeline.models import (
    Animal,
    Base,
    Camera,
    Feature,
    FeatureTracking,
    TrackedAnimal,
    Tracking,
    VideoClip,
)


@pytest.fixture(scope="module")
def engine() -> Engine:
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def db_session(engine: Engine) -> Generator[Session, None, None]:
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session_with_data(db_session: Session) -> Session:
    db_session.add_all(
        [
            Camera(id=1, name="camera1", latitude=0.0, longitude=0.0),
            Camera(id=2, name="camera2", latitude=0.0, longitude=0.0),
        ]
    )
    db_session.add_all(
        [
            Animal(id=1, name="generic"),
            Animal(id=2, name="silverback"),
            Animal(id=3, name="infant"),
        ]
    )
    db_session.add_all(
        [
            Feature(id=1, feature_name="face 90 degrees"),
            Feature(id=2, feature_name="face "),
            Feature(id=3, feature_name="foot"),
        ]
    )
    generic_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
    db_session.add_all(
        [
            VideoClip(
                id=1,
                camera_id=1,
                start_time=generic_date,
                fps=30,
                frames=1000,
                width=1920,
                height=1080,
                grayscale=False,
            ),
            VideoClip(
                id=2,
                camera_id=2,
                start_time=generic_date,
                fps=60,
                frames=1000,
                width=1920,
                height=1080,
                grayscale=False,
            ),
        ]
    )
    db_session.add_all(
        [
            TrackedAnimal(id=1, video_id=1, animal_type=1),
            TrackedAnimal(id=2, video_id=1, animal_type=2),
            TrackedAnimal(id=3, video_id=2, animal_type=3),
        ]
    )

    generic_bbox = {
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.5,
        "height": 0.5,
        "confidence": 0.5,
    }

    db_session.add_all(
        [
            Tracking(tracked_animal_id=1, frame_number=0, **generic_bbox),
            Tracking(tracked_animal_id=1, frame_number=60, **generic_bbox),
            Tracking(tracked_animal_id=3, frame_number=0, **generic_bbox),
            Tracking(tracked_animal_id=3, frame_number=60, **generic_bbox),
        ]
    )

    db_session.add_all(
        [
            FeatureTracking(
                tracked_animal_id=1,
                frame_number=0,
                feature_id=1,
                **generic_bbox,
            ),
            FeatureTracking(
                tracked_animal_id=1,
                frame_number=60,
                feature_id=2,
                **generic_bbox,
            ),
            FeatureTracking(
                tracked_animal_id=3,
                frame_number=0,
                feature_id=1,
                **generic_bbox,
            ),
            FeatureTracking(
                tracked_animal_id=3,
                frame_number=60,
                feature_id=2,
                **generic_bbox,
            ),
        ]
    )

    return db_session


def test_timedelta_video_property(db_session_with_data: Session) -> None:
    videos = db_session_with_data.query(VideoClip)
    video_1 = videos.filter(VideoClip.id == 1).one()
    assert video_1.duration == datetime.timedelta(seconds=1000 / 30)
    video_2 = videos.filter(VideoClip.id == 2).one()
    assert video_2.duration == datetime.timedelta(seconds=1000 / 60)


def test_timedelta_tracking_duration_property(db_session_with_data: Session) -> None:
    tracked_animals = db_session_with_data.query(TrackedAnimal)
    tracked_animal_1 = tracked_animals.filter(TrackedAnimal.id == 1).one()
    assert tracked_animal_1.tracking_duration == datetime.timedelta(seconds=60 / 30)
    tracked_animal_3 = tracked_animals.filter(TrackedAnimal.id == 3).one()
    assert tracked_animal_3.tracking_duration == datetime.timedelta(seconds=60 / 60)
