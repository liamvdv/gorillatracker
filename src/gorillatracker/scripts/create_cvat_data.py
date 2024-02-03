import os
import random

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def save_random_frame(video_path: str, output_dir: str) -> str:
    """
    Get the path of one random frame from a video and save it as an image.

    Args:
        video_path: path to video file
        output_dir: directory to save frames
    Returns:
        image_path: path to the saved image
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    random_frame_number = random.randint(0, int(total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    success, image = cap.read()
    if not success:
        raise ValueError("Failed to read frame from video")
    cap.release()
    os.makedirs(output_dir, exist_ok=True)
    image_name = f"{os.path.basename(video_path).split('.')[0]}_{random_frame_number}.png"
    image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(image_path, image)
    return image_path


def save_yolo_annotation(image_path: str, output_dir: str, yolo_model: YOLO) -> bool:
    """
    Save the annotation created by YOLO model for a given image.

    Args:
        image_path: path to the image
        output_dir: directory to save bounding box information
        yolo_model: YOLO model
        
    Returns:
        bool: True if the annotation is not empty, False otherwise
    """
    result = yolo_model(image_path)
    annotation_file = os.path.basename(image_path).replace(".png", ".txt")
    annotation_path = os.path.join(output_dir, annotation_file)
    if len(result) == 0:
        return False
    result[0].save_txt(annotation_path, save_conf=False)
    return True


def process_videos(video_paths: list[str], output_dir: str, yolo_model: YOLO, samples: int) -> None:
    """
    Process multiple videos to extract random frames and save their annotations.

    Args:
        video_paths: list of paths to video files
        output_dir: directory to save frames and annotations
        yolo_model: YOLO model
        samples: number of random frames to extract from each video
    """
    image_paths = []
    for video_path in tqdm(video_paths, desc="Processing videos", unit="video", total=len(video_paths)):
        for _ in range(samples):
            image_path = save_random_frame(video_path, output_dir + "/images")
            has_anno = save_yolo_annotation(image_path, output_dir + "/obj_train_data", yolo_model)
            if has_anno:
                image_paths.append(image_path)
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for image_path in image_paths:
            f.write(f"data/obj_train_data/{os.path.basename(image_path)}\n")


if __name__ == "__main__":
    video_dir = "/workspaces/gorillatracker/video_data"
    videos_path = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_cvat_data/selected_clips.txt"
    videos = open(videos_path, "r").read().splitlines()
    video_paths = [os.path.join(video_dir, file) for file in videos]
    output_dir = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_cvat_data/face"
    yolo_model_path = "/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/face.pt"
    yolo_model = YOLO(yolo_model_path)
    process_videos(video_paths, output_dir, yolo_model, 1)
