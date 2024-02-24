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
                file_path="path/to/video1",
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
                file_path="path/to/video2",
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
