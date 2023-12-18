import cv2
import os
import json
from gorillatracker.scripts.video_json_tracker import GorillaVideoTracker
from typing import Dict

def crop_and_save_image(frame, x: float, y: float, w: float, h: float, output_path: str) -> None:
    """Crop the image at the given path using the given bounding box coordinates and save it to the given output path.

    Args:
        image_path: Path to the image to crop.
        x: Relative x coordinate of the center of the bounding box.
        y: Relative y coordinate of the center of the bounding box.
        w: Relative width of the bounding box.
        h: Relative height of the bounding box.
        output_path: Path to save the cropped image to.
    """

    # calculate the bounding box coordinates
    frame_height, frame_width, _ = frame.shape
    left = int((x - (w / 2)) * frame_width)
    right = int((x + (w / 2)) * frame_width)
    top = int((y - (h / 2)) * frame_height)
    bottom = int((y + (h / 2)) * frame_height)
    
    cropped_frame = frame[top:bottom, left:right]
    cv2.imwrite(output_path, cropped_frame)
    
    
def create_dataset_from_videos(video_path: str, json_path: str, output_dir: str) -> None:
    """Create a dataset of cropped images from the video in the given path.

    Args:
        video_path: Path to the video.
        output_dir: Path to the directory to save the cropped images to.
    """ 
    
    images_per_individual = 15
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    id_frames = get_frames_for_ids(json_path)
    # open the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(video_name)
    video = cv2.VideoCapture(video_path)

    for id, frames in id_frames.items():
        if(len(frames) < images_per_individual):
            continue
        step_size = len(frames)//images_per_individual
        frame_list = [frames[i] for i in range(0, images_per_individual * step_size, step_size)]
        for frame_idx, bbox in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame = video.read()[1]
            crop_and_save_image(frame, bbox[0], bbox[1], bbox[2], bbox[3], os.path.join(output_dir, f"{video_name}-{id}-{frame_idx}.jpg"))
    video.release()
    

def get_frames_for_ids(json_path: str) -> dict[int, list[(int, (float, float, float, float))]]:
    """Get the frames for the given IDs.

    Args:
        json_path: Path to the JSON file containing the IDs.

    Returns:
        A list of lists of frames for each ID.
    """
    id_frames: Dict[int, list[(int, (float, float, float, float))]] = {}
    face_class: int = 0
    # read the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame_idx, frame in enumerate(data["labels"]):
        for bbox in frame:
            if bbox["class"] != face_class:
                continue
            id = int(bbox["id"])
            if id not in id_frames:
                id_frames[id] = []
            id_frames[id].append((frame_idx,(bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"])))
                        
    return id_frames
    
#get_frames_for_ids("/workspaces/gorillatracker/tmp/R014_20220628_151_tracked.json")
create_dataset_from_videos("/workspaces/gorillatracker/videos/R014_20220628_151.mp4", "/workspaces/gorillatracker/tmp/R014_20220628_151_tracked.json", "/workspaces/gorillatracker/tmp")   