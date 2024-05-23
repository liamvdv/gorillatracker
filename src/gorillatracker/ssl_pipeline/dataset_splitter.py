import src.gorillatracker.ssl_pipeline.video_filter_queries as vq
from sqlalchemy import create_engine
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker
from gorillatracker.ssl_pipeline.models import Video
from sqlalchemy import Select
import json
from typing import Literal
from simple_parsing import field


@dataclass(kw_only=True) 
class SplitArgs:
    db_uri: str
    version: str
    name: str = field(default = "SSL-Video-Split")
    split_by: Literal["percentage",
                      "cameras",
                      "custom"] = field(default="split")
    
    train_split: int = field(default=80)
    val_split: int = field(default=10)
    test_split: int = field(default=10)
    
    train_years: list[int] = field(default=list(range(2010,2030)))
    val_years: list[int] = field(default=list(range(2010,2030)))
    test_years: list[int] = field(default=list(range(2010,2030)))
    
    train_months: list[int] = field(default=list(range(1,13)))
    val_months: list[int] = field(default=list(range(1,13)))
    test_months: list[int] = field(default=list(range(1,13)))
    
    hours: list[int] = field(default=list(range(0,24)))
    video_length: tuple[int,int] = field(default=(0,1000000)) # min, max video length in seconds
    min_videos_per_camera: int = field(default=0)
    
    max_train_videos: int = field(default=1000000)
    max_val_videos: int = field(default=1000000)
    max_test_videos: int = field(default=1000000)

def build_train_query(args: SplitArgs) -> Select[tuple[Video]]:
    query = vq.get_video_query(args.version)
    query = vq.filter_by_years(query, args.train_years)
    query = vq.filter_by_months(query, args.train_months)
    query = vq.filter_by_hours(query, args.hours)
    query = vq.filter_by_video_length(query, args.video_length)
    query = vq.filter_by_min_videos_per_camera(query, args.min_videos_per_camera)
    query = query.limit(args.max_train_videos)
    return query

def build_val_query(args: SplitArgs) -> Select[tuple[Video]]:
    query = vq.get_video_query(args.version)
    query = vq.filter_by_years(query, args.val_years)
    query = vq.filter_by_months(query, args.val_months)
    query = vq.filter_by_hours(query, args.hours)
    query = vq.filter_by_video_length(query, args.video_length)
    query = vq.filter_by_min_videos_per_camera(query, args.min_videos_per_camera)
    query = query.limit(args.max_val_videos)
    return query

def build_test_query(args: SplitArgs) -> Select[tuple[Video]]:
    query = vq.get_video_query(args.version)
    query = vq.filter_by_years(query, args.test_years)
    query = vq.filter_by_months(query, args.test_months)
    query = vq.filter_by_hours(query, args.hours)
    query = vq.filter_by_video_length(query, args.video_length)
    query = vq.filter_by_min_videos_per_camera(query, args.min_videos_per_camera)
    query = query.limit(args.max_test_videos)
    return query

def split_by_percentage(args: SplitArgs) -> tuple[list[Video], list[Video], list[Video]]:
    assert args.train_split + args.val_split + args.test_split == 100, "The sum of the split percentages must be 100"
    query = build_train_query(args)
    engine = create_engine(args.db_uri)
    session = sessionmaker(bind=engine)
    videos = vq.get_videos_from_query(query, session())
    train_end = int(len(videos) * args.train_split / 100)
    val_end = train_end + int(len(videos) * args.val_split / 100)
    train = videos[:train_end]
    val = videos[train_end:val_end]
    test = videos[val_end:]
    return train, val, test

def split_by_cameras(args: SplitArgs) -> tuple[list[Video], list[Video], list[Video]]:
    assert args.train_split + args.val_split + args.test_split == 100, "The sum of the split percentages must be 100"
    engine = create_engine(args.db_uri)
    session = sessionmaker(bind=engine)
    cameras = vq.get_camera_ids(session())
    train_cameras = cameras[:int(len(cameras) * args.train_split / 100)]
    val_cameras = cameras[len(train_cameras):len(train_cameras) + int(len(cameras) * args.val_split / 100)]
    test_cameras = cameras[len(train_cameras) + len(val_cameras):]
    train_videos = vq.get_videos_from_query(vq.camera_id_filter(build_train_query(args), train_cameras), session())
    val_videos = vq.get_videos_from_query(vq.camera_id_filter(build_val_query(args), val_cameras), session())
    test_videos = vq.get_videos_from_query(vq.camera_id_filter(build_test_query(args), test_cameras), session())
    return train_videos, val_videos, test_videos

def split_custom(args: SplitArgs) -> tuple[list[Video], list[Video], list[Video]]:
    engine = create_engine(args.db_uri)
    session = sessionmaker(bind=engine)
    train = vq.get_videos_from_query(build_train_query(args), session())
    val = vq.get_videos_from_query(build_val_query(args), session())
    test = vq.get_videos_from_query(build_test_query(args), session())
    val = [video for video in val if video not in train]
    test = [video for video in test if video not in train and video not in val]
    return train, val, test

def write_split_to_json(train: list[Video], val: list[Video], test: list[Video], args:SplitArgs) -> None:
    """Write the split to a json file."""
    with open(f"{args.name}.json", "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f)
              
if __name__ == "__main__":
    args = SplitArgs(db_uri="db-uri-here", version="2024-04-18")
    args.name = f"{args.name}_{args.version}_{args.split_by}_split"
    if(args.split_by == "percentage"):
        train, val, test = split_by_percentage(args)
    elif(args.split_by == "cameras"):
        train, val, test = split_by_cameras(args)
    elif(args.split_by == "custom"):
        train, val, test = split_custom(args)
    else:
        raise ValueError("Invalid split_by argument")
    write_split_to_json(train, val, test, args)