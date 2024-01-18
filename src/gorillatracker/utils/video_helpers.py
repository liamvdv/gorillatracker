import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

from gorillatracker.type_helper import BoundingBox
from gorillatracker.utils.yolo_helpers import convert_from_yolo_format


GORILLA = 0
GORILLA_FACE = 1

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


@dataclass
class TrackedFrame:
    frame: int
    bounding_box: BoundingBox
    confidence: float


@dataclass
class TrackedGorilla:
    video_id: str
    individual_id: int
    bounding_boxes: List[TrackedFrame] = field(default_factory=list)
    bounding_boxes_face: List[TrackedFrame] = field(default_factory=list)

    def start_frame(self) -> int:
        return self.bounding_boxes[0].frame

    def end_frame(self) -> int:
        return self.bounding_boxes[-1].frame


@dataclass
class VideoClip:
    """
    A 30 second recording of a gorillas
    """

    video_id: str
    camera_id: str
    video_path: str
    start_time: datetime
    trackings: List[TrackedGorilla] = field(default_factory=list)


@dataclass
class Video:
    """
    Ensamble of video clips from a gorilla group infront of a camera
    """

    camera_id: str
    subclips: List[VideoClip] = field(default_factory=list)


# parse tracked video from json
def _parse_tracked_gorillas(video_id: str, json: str) -> List[TrackedGorilla]:
    tracked_gorillas = {int(i["id"]): TrackedGorilla(video_id, int(i["id"])) for i in json["tracked_IDs"]}  # type: ignore
    for frame_n, frame in enumerate(json["labels"]):  # type: ignore
        for yolo_bbox in frame:
            tracked_gorilla = tracked_gorillas[int(yolo_bbox["id"])]  # type: ignore
            bbox = convert_from_yolo_format(
                list(map(float, (yolo_bbox["center_x"], yolo_bbox["center_y"], yolo_bbox["w"], yolo_bbox["h"]))),  # type: ignore
                VIDEO_WIDTH,
                VIDEO_HEIGHT,
            )
            tracked_frame = TrackedFrame(frame_n, bbox, float(yolo_bbox["conf"]))  # type: ignore

            class_id = int(yolo_bbox["class"])  # type: ignore
            assert class_id in [GORILLA, GORILLA_FACE]
            if class_id == GORILLA:
                tracked_gorilla.bounding_boxes.append(tracked_frame)
            elif class_id == GORILLA_FACE:
                tracked_gorilla.bounding_boxes_face.append(tracked_frame)

    return list(tracked_gorillas.values())


def _parse_tracked_video(path: str, timestamps: Dict[str, str]) -> VideoClip:
    filename, _ = os.path.splitext(os.path.basename(path))
    filename = filename[: -len("_tracked")]
    camera_id, date_str, _ = filename.split("_")
    date = datetime.strptime(date_str, "%Y%m%d")
    video_clip_json = json.load(open(path))
    timestamp = timestamps[filename]
    daytime = datetime.strptime(timestamp, "%I:%M %p")
    date = datetime.combine(date, daytime.time())
    tracked_gorillas = _parse_tracked_gorillas(filename, video_clip_json)
    return VideoClip(filename, camera_id, path, date, tracked_gorillas)


# parse folders with the json of tracked videos
def parse_dataset(path: str, timestamps_path: str) -> List[Video]:
    timestamps = json.load(open(timestamps_path))
    for video_clip_filename in os.listdir(path):
        if video_clip_filename == ".mp4_tracked.json":
            continue
        assert video_clip_filename.endswith(".json")
        assert len(video_clip_filename.split("_")) == 4
        video_clip = _parse_tracked_video(os.path.join(path, video_clip_filename), timestamps)

    return []


if __name__ == "__main__":
    path = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels_tracked"
    timestamps = "/workspaces/gorillatracker/data/derived_data/timestamps.json"
    # TODO assert video width and height
    parse_dataset(path, timestamps)
