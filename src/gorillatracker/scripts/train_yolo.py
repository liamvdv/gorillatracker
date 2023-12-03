"""This file contains short scripts to train a yolo model using the bristol dataset and detect gorilla faces in the CXL dataset."""

import os
import shutil
import time
import logging

from ultralytics import YOLO
from typing import Tuple, Any, Literal

model_paths = {
    "yolov8n": "/workspaces/gorillatracker/yolov8n.pt",
    "yolov8m": "/workspaces/gorillatracker/yolov8m.pt",
    "yolov8x": "/workspaces/gorillatracker/yolov8x.pt",
}

logger = logging.getLogger(__name__)


def train_yolo(model_name: Literal["yolov8n", "yolov8m", "yolov8x"], epochs: int, batch_size: int, dataset_yml: str, wandb_project: str) -> Tuple[YOLO, Any]:
    """Train a YOLO model with the given parameters.

    Args:
        model_name (str): Name of the yolo model to train.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size to use.
        dataset_yml (str): Path to the dataset yml file.
        wandb_project (str): Name of the wandb project to use.

    Returns:
        ultralytics.YOLO: Trained yolo model.
        dict: Training result. See ultralytics docs for details.
    """

    model = YOLO(model_paths[model_name])
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    training_name = f"{model_name}-e{epochs}-b{batch_size}-{timestamp}"

    logger.info("Training model %s with %d epochs and batch size of %d", model_name, epochs, batch_size)

    result = model.train(
        name=training_name, data=dataset_yml, epochs=epochs, batch=batch_size, patience=10, project=wandb_project
    )

    
    shutil.move(wandb_project, f"logs/{wandb_project}-{training_name}")
    logger.info("Training finished for %s. Results in logs/%s-%s", training_name, wandb_project, training_name)
    return model, result


def set_annotation_class_0(annotation_dir: str, dest_dir: str) -> None:
    """Set the class of all annotations to 0 (gorilla face) and save them in the destination directory.

    Args:
        annotation_dir (str): Directory containing the annotation files.
        dest_dir (str): Directory to save the new annotation files to.
        file_extension (str, optional): File extension of the images. Defaults to ".jpg".
    """
    for annotation_filename in filter(lambda f: f.endswith(".txt"), os.listdir(annotation_dir)):
        with open(os.path.join(annotation_dir, annotation_filename)) as annotation_file:
            new_lines = ["0 " + " ".join(line.strip().split(" ")[1:]) for line in annotation_file if line.strip()]

        with open(os.path.join(dest_dir, annotation_filename), "w") as new_annotation_file:
            new_annotation_file.write("\n".join(new_lines))


def join_annotations_and_imgs(image_dir: str, annotation_dir: str, output_dir: str, file_extension: str =".jpg") -> None:
    """Build a dataset for yolo using the given image and annotation directories.

    Args:
        image_dir (str): Directory containing the images.
        annotation_dir (str): Directory containing the annotation files.
        output_dir (str): Directory to merge the images and annotations into.
        file_extension (str, optional): File extension of the images. Defaults to ".png".
    """
    image_files = os.listdir(image_dir)
    image_files = list(filter(lambda x: x.endswith(file_extension), image_files))

    for image_file in image_files:
        annotation_file = image_file.replace(file_extension, ".txt")
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        assert os.path.exists(annotation_path), f"Annotation file {annotation_path} does not exist"

        shutil.copyfile(annotation_path, os.path.join(output_dir, annotation_file))
        if not os.path.exists(os.path.join(output_dir, image_file)):
            shutil.copyfile(os.path.join(image_dir, image_file), os.path.join(output_dir, image_file))


def remove_files_from_dir_with_extension(annotation_dir: str, file_extension: str = ".txt") -> None:
    """Remove all files ending with the given file extension from the given directory."""

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith(file_extension):
            os.remove(os.path.join(annotation_dir, annotation_file))


def detect_gorillafaces_cxl(model: YOLO, image_dir: str, output_dir: str, file_extension: str = ".png") -> None:
    """Detect gorilla faces in the given directory and save the results in the output directory using the given yolo model."""
    image_files = os.listdir(image_dir)
    image_files = list(filter(lambda x: x.endswith(file_extension), image_files))

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = model(image_path)

        annotation_path = os.path.join(output_dir, image_file.replace(file_extension, ".txt"))
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        result[0].save_txt(annotation_path, save_conf=True)  # NOTE: simply appends to the .txt file
