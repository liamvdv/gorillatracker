import src.gorillatracker.ssl_pipeline.video_filter_queries as vq
from sqlalchemy import create_engine
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker
from gorillatracker.ssl_pipeline.models import Video
from sqlalchemy.orm import Session
from sqlalchemy import Select
import json
import argparse
from typing import List, Literal, Union
from simple_parsing import field, list_field
from simple_parsing import parse


@dataclass(kw_only=True) 
class SplitArgs:
    db_uri: str
    version: str
    name: str = field(default = "SSL-Video-Split")
    split_by_percentage: bool = field(default=True)
    
    train_split: int = field(default=80)
    val_split: int = field(default=10)
    test_split: int = field(default=10)
    
    train_years: list[int] = field(default=list(range(2010,2030)))
    val_years: list[int] = field(default=list(range(2010,2030)))
    test_years: list[int] = field(default=list(range(2010,2030)))
    
    train_months: list[int] = field(default=list(range(0,13)))
    val_months: list[int] = field(default=list(range(0,13)))
    test_months: list[int] = field(default=list(range(0,13)))
    
    hours: list[int] = field(default=list(range(0,24)))
    video_length: tuple[int,int] = field(default=(0,1000000)) # min, max video length in seconds

def build_query(args: argparse.Namespace) -> Select[tuple[Video]]:
    query = vq.get_video_query(args.version)
    # put args check here
    return query
    

def build_filter_query(args: argparse.Namespace) -> Select[tuple[Video]]:
    query = vq.get_video_query(args.version)
    # put args check here
    return query

def split_by_filter(query: Select[tuple[Video]], filter_query: Select[tuple[Video]], session: Session) -> tuple[list[Video], list[Video], list[Video]]:
    """creates a split of videos based on a filter query. The filter query is used to filter the videos for train.
    The remaing videos are val. test is empty"""
    all_videos = vq.get_videos_from_query(query, session)
    filtered_videos = vq.get_videos_from_query(filter_query, session)
    train = [video.video_id for video in all_videos if video in filtered_videos]
    val = [video.video_id for video in all_videos if video not in filtered_videos]
    test = []
    return train, val, test

def split_by_percentage(train:int, val:int, test:int, query: Select[tuple[Video]], session: Session) -> tuple[list[Video], list[Video], list[Video]]:
    """Split the videos into train, validation, and test sets by percentage."""
    assert train + val + test == 100, "The sum of the split percentages must be 100"
    result = session.execute(query).scalars().all()
    videos = [video.video_id for video in result]
    num_videos = len(videos)
    train_end = int(num_videos * train / 100)
    val_end = train_end + int(num_videos * val / 100)
    train = videos[:train_end]
    val = videos[train_end:val_end]
    test = videos[val_end:]
    return train, val, test

def write_split_to_json(train: list[Video], val: list[Video], test: list[Video], args:SplitArgs) -> None:
    """Write the split to a json file."""
    with open(f"{args.name}.json", "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f)
              
if __name__ == "__main__":
    config_path = "./cfgs/config.yml"
    args = parse(SplitArgs, config_path=config_path)
    assert args.db_uri is not None, "Please provide a db uri"
    assert args.version is not None, "Please provide a version"
    args.name = f"{args.name}_{args.version}"
    
    engine = create_engine(args.db_uri)
    session = sessionmaker(bind=engine)
    
    query = vq.get_video_query(args.version)
    filter_query = getattr(vq, args.filter)(query, args.split)
    train, val, test = split_by_filter(query, filter_query, session)
    write_split_to_json(train, val, test, args)