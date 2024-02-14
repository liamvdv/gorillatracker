import json
import tempfile
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2
from typing import Dict, List, Union
from ultralytics import YOLO
import pandas as pd

from gorillatracker.scripts.video_json_tracker import GorillaVideoTracker
from gorillatracker.utils.embedding_generator import generate_embeddings_from_tracked_video, get_model
from gorillatracker.transform_utils import SquarePad


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
        face_model_path = "/workspaces/gorillatracker/models/yolov8n_gorillaface_knwmwxko.pt"
        precict_video_simple(video_path, json_path, YOLO(face_model_path), YOLO(body_model_path), {})

        tracker = GorillaVideoTracker(path=temp_dir)
        tracker.track_file(json_path)
        # read out json in out_path directory with tracked
        tracked_json_path = f"{temp_dir}/{file_name}_tracked.json"
        with open(tracked_json_path, "r") as file:
            tracked_json = json.load(file)
        tracked_json = tracked_json["labels"]
        return tracked_json

    
def get_frames_for_ids(data: str):
    id_frames = {}
    for frame_idx, frame in enumerate(data):
        for bbox in frame:
            if bbox["class"] != 1:  # faceclass
                continue
            id = int(bbox["id"])
            if id not in id_frames:
                id_frames[id] = []
            id_frames[id].append((frame_idx, (bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"])))
    return id_frames


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
    for frame in face_results:
        boxes = frame.boxes.xywhn.tolist()
        confs = frame.boxes.conf.tolist()
        frame = []
        for box, conf in zip(boxes, confs):
            x, y, w, h = box
            box = {"class": 1,"center_x": x, "center_y": y, "w": w, "h": h, "conf": conf}
            labeled_video_frames[index].append(box)
        index += 1

    json.dump({"labels":labeled_video_frames}, open(json_path, "w"), indent=4)


def convert_tracked_list_to_df(data: Dict, class_id=1):
    tracked_df = pd.DataFrame(columns=["frame_id", "bbox", "individual_id"])
    for frame_idx, frame in enumerate(data):
        for bbox in frame:
            if bbox["class"] != class_id:
                continue
            tracked_df = pd.concat(
                [
                    tracked_df,
                    pd.DataFrame(
                        {
                            "individual_id": [bbox["id"]],
                            "frame_id": [frame_idx],
                            "bbox": [(bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"])],
                        }
                    ),
                ],
                ignore_index=True,
            )
    tracked_df = tracked_df.sort_values(by=["frame_id", "individual_id"])
    tracked_df.set_index(["frame_id", "individual_id"], inplace=True)
    tracked_df.rename(columns={"bbox": "body_bbox"}, inplace=True)
    return tracked_df


def get_tracking_and_embedding_data_for_video(video_path: str, model_from_run: str = "https://wandb.ai/gorillas/Embedding-SwinV2Large-CXL-Open/runs/a4t93htr/overview"):
    """Generates a DataFrame with:
    Index: frame_id, individual_id
    Columns: body_bbox, face_bbox, face_embedding
    """
    tracked_video_data = create_tracked_bbox_json(video_path)
    
    # Examples usage with embedding generation
    model = get_model(model_from_run)
    model_transforms = transforms.Compose(
        [
            SquarePad(),
            transforms.ToTensor(),
            # TODO(liamvdv: Add more transforms here if needed?)
            model.get_tensor_transforms(),
        ]
    )
    
    tracked_video_frames = get_frames_for_ids(tracked_video_data)
    face_frame_embedding_df = generate_embeddings_from_tracked_video(model, video_path, tracked_video_frames, model_transforms)
    
    body_frame_df = convert_tracked_list_to_df(tracked_video_data, class_id=0)
    final_df = body_frame_df.join(face_frame_embedding_df, on=["frame_id", "individual_id"])
    final_df.reset_index(inplace=True)
    return final_df


if __name__ == "__main__":
    video_path = "/workspaces/gorillatracker/video_data/R506_20220330_184.mp4"
    df = get_tracking_and_embedding_data_for_video(video_path)
    
    print(df.head())

