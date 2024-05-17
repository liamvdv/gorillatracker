import datetime as dt
from pathlib import Path
from typing import Iterator, Optional, Sequence

from sqlalchemy import ColumnElement, Select, alias, func, or_, select
from sqlalchemy.orm import Session, aliased

from gorillatracker.ssl_pipeline.models import (
    Camera,
    Task,
    TaskStatus,
    TaskType,
    Tracking,
    TrackingFrameFeature,
    Video,
    VideoFeature,
)

# filter video count

# filter camera count

# filter camera with x videos

# filter for year

# filter for time

# filter for months