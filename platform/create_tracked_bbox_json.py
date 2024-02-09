import json
import tempfile
import os 
from pathlib import Path
from typing import Dict, List, Optional, Union

import ultralytics
from ultralytics import YOLO
import tempfile

from gorillatracker.scripts.video_json_tracker import GorillaVideoTracker

def create_tracked_bbox_json(video_path: str):
    """Create tracked Bounding Boxes json.

    Args:
        video_path: Path to the video.
        out_path: Path to save the JSON file to.

    Returns:
        List of Bounding Boxes.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = Path(video_path).stem
        json_path = f"{temp_dir}/{file_name}.json"
        body_model_path = "/workspaces/gorillatracker/models/yolov8n_gorillabody_ybyh495y.pt"
        face_model_path = "/workspaces/gorillatracker/models/yolov8n_gorillaface_a2mkg0zc.pt"
        precict_video_simple(video_path, json_path, YOLO(face_model_path), YOLO(body_model_path), {})

        tracker = GorillaVideoTracker(path=temp_dir)
        tracker.track_file(json_path)
        # read out json in out_path directory with tracked
        tracked_json_path = f"{temp_dir}/{file_name}_tracked.json"
        with open(tracked_json_path, "r") as file:
            tracked_json = json.load(file)
        tracked_json = tracked_json["labels"]
        return tracked_json

def precict_video_simple(
        video_path: str,
        json_path: str,
        face_model: YOLO,
        body_model: YOLO,
        yolo_args: Dict[str, Union[bool, int, str]] = {}
) -> None:
    """
    Predicts labels for objects in a video using a YOLO body and face model

    Parameters:
    - video_path (str): The path to the input video file.
    - json_path (str): The path to the output JSON file.
    - model (YOLO): The YOLO model to use for prediction.
    - yolo_args (Dict): Additional arguments to pass to the YOLO model.
    Returns:
        None
    """

    face_results = face_model.predict(video_path, stream=True, **yolo_args)
    body_results = body_model.predict(video_path, stream=True, **yolo_args)
    labeled_video_frames: List[List[Dict[str, float]]] = []
    #body
    for frame in body_results:
        boxes = frame.boxes.xywhn.tolist()
        confs = frame.boxes.conf.tolist()
        frame = []
        for box, conf in zip(boxes, confs):
            x, y, w, h = box
            box = {"class": 0,"center_x": x, "center_y": y, "w": w, "h": h, "conf": conf}
            frame.append(box)
        labeled_video_frames.append(frame)
    #face
    index = 0
    for frame in body_results:
        boxes = frame.boxes.xywhn.tolist()
        confs = frame.boxes.conf.tolist()
        frame = []
        for box, conf in zip(boxes, confs):
            x, y, w, h = box
            box = {"class": 1,"center_x": x, "center_y": y, "w": w, "h": h, "conf": conf}
            labeled_video_frames[index].append(box)
        index += 1

    json.dump({"labels":labeled_video_frames}, open(json_path, "w"), indent=4)


# main 
if __name__ == "__main__":
    video_path = "/workspaces/gorillatracker/video_data/R506_20220330_184.mp4"
    
    json_data = create_tracked_bbox_json(video_path)
    # print out first frame
    print(json_data[:1])

