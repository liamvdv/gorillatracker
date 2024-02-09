import os
import random

import cv2
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


def save_yolo_annotation(image_path: str, output_dir: str, yolo_model: YOLO) -> None:
    """
    Save the annotation created by YOLO model for a given image.

    Args:
        image_path: path to the image
        output_dir: directory to save bounding box information
        yolo_model: YOLO model
    """
    result = yolo_model(image_path)
    annotation_file = os.path.basename(image_path).replace(".png", ".txt")
    annotation_path = os.path.join(output_dir, annotation_file)
    result[0].save_txt(annotation_path, save_conf=False)


def process_videos(video_paths: list[str], output_dir: str, yolo_model: YOLO) -> None:
    """
    Process multiple videos to extract random frames and save their annotations.

    Args:
        video_paths: list of paths to video files
        output_dir: directory to save frames and annotations
        yolo_model: YOLO model
    """
    image_paths = []
    for video_path in video_paths:
        image_path = save_random_frame(video_path, output_dir + "/images")
        save_yolo_annotation(image_path, output_dir + "/annotations", yolo_model)
        image_paths.append(image_path)
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for image_path in image_paths:
            f.write(f"{image_path}\n")


if __name__ == "__main__":
    video_dir = "/workspaces/gorillatracker/video_data"
    video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]
    output_dir = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_cvat_data"
    yolo_model_path = "/workspaces/gorillatracker/models/yolov8n_gorillabody_ybyh495y.pt"
    yolo_model = YOLO(yolo_model_path)
    process_videos(video_paths[0:10], output_dir, yolo_model)
