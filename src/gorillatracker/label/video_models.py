import datetime as dt
import json
import os
from datetime import datetime
from itertools import groupby
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from gorillatracker.scripts.cutout import calculate_area
from gorillatracker.type_helper import BoundingBox
from gorillatracker.utils.yolo_helpers import convert_from_yolo_format


class TrackerType:
    """
    As defined in our json files.
    """

    GORILLA = 0
    GORILLA_FACE = 1


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FRAMES_PER_SECOND = 60

EPSILON = 0.0001


class TrackedFrame(BaseModel):
    """
    A bounding box in a frame.
    """

    frame: int
    bounding_box: BoundingBox
    confidence: float

    def bbox_size(self) -> float:
        return calculate_area(self.bounding_box)


class TrackedGorilla(BaseModel):
    """
    A gorilla that is tracked in a video clip.
    """

    video_id: str
    individual_id: int
    negative_ids: List[int] = Field(default_factory=list)
    bounding_boxes: List[TrackedFrame] = Field(default_factory=list)
    bounding_boxes_face: List[TrackedFrame] = Field(default_factory=list)

    def start_frame(self) -> int:
        return min([bbox.frame for bbox in self.bounding_boxes])

    def end_frame(self) -> int:
        return max([bbox.frame for bbox in self.bounding_boxes])

    def avg_bbox_size(self) -> float:
        return sum([bbox.bbox_size() for bbox in self.bounding_boxes]) / (len(self.bounding_boxes) + EPSILON)

    def bboxes_above_size(self, size: float) -> int:
        return len([bbox for bbox in self.bounding_boxes if bbox.bbox_size() > size])


class VideoClip(BaseModel):
    """
    A single mp4 file that is recorded by a camera.
    """

    video_id: str
    camera_id: str
    start_time: datetime
    total_frames: int = Field(default=0)
    trackings: List[TrackedGorilla] = Field(default_factory=list)

    @property
    def end_time(self) -> datetime:
        return self.start_time + dt.timedelta(seconds=self.total_frames / FRAMES_PER_SECOND)

    def avg_bbox_size(self) -> float:
        return sum([tracking.avg_bbox_size() for tracking in self.trackings]) / (len(self.trackings) + EPSILON)

    def bboxes_above_size(self, size: float) -> int:
        return sum([tracking.bboxes_above_size(size) for tracking in self.trackings])

    def tracked_gorillas(self) -> int:
        return len(self.trackings)

    def tracked_gorilla_frames(self, tracker: TrackerType) -> int:
        return sum([len(tracking.bounding_boxes) for tracking in self.trackings])

    def __repr__(self) -> str:
        return f"VideoClip(video_id={self.video_id}, camera_id={self.camera_id}, start_time={self.start_time}, trackings_len={len(self.trackings)})"


class Video(BaseModel):
    """
    A video is a collection of VideoClips, as the cameras recording stop after a certain amount of time.
    """

    camera_id: str
    subclips: List[VideoClip] = Field(default_factory=list)

    @property
    def start_time(self) -> datetime:
        return min([clip.start_time for clip in self.subclips])

    @property
    def end_time(self) -> datetime:
        return max([clip.end_time for clip in self.subclips])

    def avg_bbox_size(self) -> float:
        return sum([clip.avg_bbox_size() for clip in self.subclips]) / (len(self.subclips) + EPSILON)

    def bboxes_above_size(self, size: float) -> int:
        return sum([clip.bboxes_above_size(size) for clip in self.subclips])

    def tracked_gorillas(self) -> int:
        return sum([clip.tracked_gorillas() for clip in self.subclips])

    def tracked_gorilla_frames(self, tracker: TrackerType) -> int:
        return sum([clip.tracked_gorilla_frames(tracker) for clip in self.subclips])


class VideoDataset(BaseModel):
    videos: List[Video]

    def avg_bbox_size(self) -> float:
        return sum([video.avg_bbox_size() for video in self.videos]) / (len(self.videos) + EPSILON)

    def bboxes_above_size(self, size: float) -> int:
        return sum([video.bboxes_above_size(size) for video in self.videos])

    def tracked_gorillas(self) -> int:
        return sum([video.tracked_gorillas() for video in self.videos])

    def tracked_gorilla_frames(self, tracker: TrackerType) -> int:
        return sum([video.tracked_gorilla_frames(tracker) for video in self.videos])


def _parse_tracked_gorilla(video_id: str, json: str) -> Tuple[int, TrackedGorilla]:
    individual_id = int(json["id"])  # type: ignore
    negative_ids = [int(i) for i in json["negatives"]]  # type: ignore
    return individual_id, TrackedGorilla(video_id=video_id, individual_id=individual_id, negative_ids=negative_ids)


# parse tracked video from json
def _parse_tracked_gorillas(video_id: str, json: str) -> List[TrackedGorilla]:
    tracked_gorillas = dict(_parse_tracked_gorilla(video_id, i) for i in json["tracked_IDs"])  # type: ignore
    for frame_n, frame in enumerate(json["labels"]):  # type: ignore
        for yolo_bbox in frame:
            tracked_gorilla = tracked_gorillas[int(yolo_bbox["id"])]  # type: ignore
            bbox = convert_from_yolo_format(
                list(map(float, (yolo_bbox["center_x"], yolo_bbox["center_y"], yolo_bbox["w"], yolo_bbox["h"]))),  # type: ignore
                VIDEO_WIDTH,
                VIDEO_HEIGHT,
            )
            tracked_frame = TrackedFrame(frame=frame_n, bounding_box=bbox, confidence=float(yolo_bbox["conf"]))  # type: ignore

            class_id = int(yolo_bbox["class"])  # type: ignore
            assert class_id in [TrackerType.GORILLA, TrackerType.GORILLA_FACE]
            if class_id == TrackerType.GORILLA:
                tracked_gorilla.bounding_boxes.append(tracked_frame)
            elif class_id == TrackerType.GORILLA_FACE:
                tracked_gorilla.bounding_boxes_face.append(tracked_frame)

    return list(tracked_gorillas.values())


def _parse_tracked_video(path: str, timestamps: Dict[str, str], parse_content: bool) -> VideoClip:
    filename, _ = os.path.splitext(os.path.basename(path))
    filename = filename[: -len("_tracked")]
    camera_id, date_str, _ = filename.split("_")
    date = datetime.strptime(date_str, "%Y%m%d")
    timestamp = timestamps[filename]
    daytime = datetime.strptime(timestamp, "%I:%M %p")
    date = datetime.combine(date, daytime.time())

    if not parse_content:
        return VideoClip(video_id=filename, camera_id=camera_id, start_time=date)

    video_clip_json = json.load(open(path))
    n_frames = len(video_clip_json["labels"])
    tracked_gorillas = _parse_tracked_gorillas(filename, video_clip_json)
    return VideoClip(
        video_id=filename, camera_id=camera_id, start_time=date, total_frames=n_frames, trackings=tracked_gorillas
    )


def _group_video_clips_by_camera_id_and_date(video_clips: List[VideoClip]) -> List[List[VideoClip]]:
    video_clips.sort(key=lambda x: (x.camera_id, x.start_time.date()))
    return [list(group) for _, group in groupby(video_clips, key=lambda x: (x.camera_id, x.start_time.date()))]


def _combine_video_clips(video_clips: List[VideoClip], video_time_difference: dt.timedelta) -> List[Video]:
    grouped_clips = _group_video_clips_by_camera_id_and_date(video_clips)
    videos = []
    for daily_clips in grouped_clips:
        daily_clips.sort(key=lambda x: x.start_time)
        video = None
        for clip in daily_clips:
            if video is None:
                video = Video(camera_id=clip.camera_id)
            elif clip.start_time - video.end_time > video_time_difference:
                videos.append(video)
                video = Video(camera_id=clip.camera_id)
            video.subclips.append(clip)
        assert video is not None
        videos.append(video)
    return videos


def parse_dataset(
    path: str, timestamps_path: str, video_time_difference: dt.timedelta, parse_content: bool
) -> VideoDataset:
    """
    Args:
        path: path to directory with tracked videos
        timestamps_path: path to timestamps.json
        video_time_difference: the max time difference until a video clip is considered a new video
        parse_content: whether to parse the content of the video clips or just the metadata
    """
    timestamps = json.load(open(timestamps_path))
    video_clips = []
    for video_clip_filename in os.listdir(path):
        if video_clip_filename == ".mp4_tracked.json":
            continue
        if not video_clip_filename.endswith(".json"):
            print(f"Skipping {video_clip_filename}")
            continue
        assert len(video_clip_filename.split("_")) == 4
        video_clip = _parse_tracked_video(os.path.join(path, video_clip_filename), timestamps, parse_content)
        video_clips.append(video_clip)
    print(len(video_clips))
    videos = _combine_video_clips(video_clips, video_time_difference)
    return VideoDataset(videos=videos)


if __name__ == "__main__":
    path = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_tracked"
    timestamps = "/workspaces/gorillatracker/data/derived_data/timestamps.json"
    # TODO(memben) assert video width and height
    video_dataset = parse_dataset(path, timestamps, dt.timedelta(minutes=10), parse_content=True)
    video_dataset_json = video_dataset.model_dump_json()
    with open("/workspaces/gorillatracker/data/derived_data/video_dataset.json", "w") as f:
        f.write(video_dataset_json)
