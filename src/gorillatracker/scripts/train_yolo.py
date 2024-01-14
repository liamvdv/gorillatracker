"""This file contains short scripts to train a yolo model using the bristol dataset and detect gorilla faces in the CXL dataset."""

import logging
import os
import shutil
import time
from typing import Any, Literal, Tuple

from ultralytics import YOLO

model_paths = {
    "yolov8n": "/workspaces/gorillatracker/yolov8n.pt",
    "yolov8m": "/workspaces/gorillatracker/yolov8m.pt",
    "yolov8x": "/workspaces/gorillatracker/yolov8x.pt",
}

logger = logging.getLogger(__name__)


def modify_dataset_train_yolo(
    bristol_split_dir: str,
    model_type: Literal["yolov8n", "yolov8m", "yolov8x"],
    epochs: int,
    batch_size: int,
    wandb_project: str = "Detection-YOLOv8-Bristol-OpenSet",
    bristol_annotation_dir: str = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox",
    bristol_yolo_annotation_dir: str = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox_class0",
    gorilla_yml_path: str = "/workspaces/gorillatracker/data/ground_truth/bristol/gorilla.yaml",
) -> YOLO:
    """Build a dataset for yolo using the bristol dataset and train a yolo model on it. When finished undo the changes to the bristol dataset.
    NOTE: The paths to the bristol dataset has to be inside the gorilla_yml_path file.

    Args:
        bristol_split_dir: Directory containing the bristol split.
        model_type: Name of the yolo model to train.
        epochs: Number of epochs to train.
        batch_size: Batch size to use.
        wandb_project: Name of the wandb project to use.
        gorilla_yml_path: Path to the gorilla yml file.
        bristol_annotation_dir: Directory containing the bristol annotations.
        bristol_yolo_annotation_dir: Directory to save the annotations for yolo.

    Returns:
        Trained yolo model."""

    # build dataset for yolo
    set_annotation_class_0(bristol_annotation_dir, bristol_yolo_annotation_dir)

    # YOLO needs the images and annotations in the same folder
    for split in ["train", "val", "test"]:
        join_annotations_and_imgs(
            os.path.join(bristol_split_dir, split), bristol_yolo_annotation_dir, os.path.join(bristol_split_dir, split)
        )

    model, _, model_name = train_yolo(model_type, epochs, batch_size, gorilla_yml_path, wandb_project=wandb_project)

    # remove annotations from the bristol split
    for split in ["train", "val", "test"]:
        remove_files_from_dir_with_extension(os.path.join(bristol_split_dir, split))

    return model, model_name


def train_yolo(
    model_name: Literal["yolov8n", "yolov8m", "yolov8x"],
    epochs: int,
    batch_size: int,
    dataset_yml: str,
    wandb_project: str,
) -> Tuple[YOLO, Any, str]:
    """Train a YOLO model with the given parameters.

    Args:
        model_name: Name of the yolo model to train.
        epochs: Number of epochs to train.
        batch_size: Batch size to use.
        dataset_yml: Path to the dataset yml file.
        wandb_project: Name of the wandb project to use.

    Returns:
        Trained yolo model.
        Training result. See ultralytics docs for details.
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
    return model, result, training_name


def set_annotation_class_0(annotation_dir: str, dest_dir: str) -> None:
    """Set the class of all annotations to 0 (gorilla face) and save them in the destination directory.

    Args:
        annotation_dir: Directory containing the annotation files.
        dest_dir: Directory to save the new annotation files to.
        file_extension: File extension of the images. Defaults to ".jpg".
    """
    for annotation_filename in filter(lambda f: f.endswith(".txt"), os.listdir(annotation_dir)):
        with open(os.path.join(annotation_dir, annotation_filename)) as annotation_file:
            new_lines = ["0 " + " ".join(line.strip().split(" ")[1:]) for line in annotation_file if line.strip()]

        with open(os.path.join(dest_dir, annotation_filename), "w") as new_annotation_file:
            new_annotation_file.write("\n".join(new_lines))


def join_annotations_and_imgs(
    image_dir: str, annotation_dir: str, output_dir: str, file_extension: str = ".jpg"
) -> None:
    """Build a dataset for yolo using the given image and annotation directories.

    Args:
        image_dir: Directory containing the images.
        annotation_dir: Directory containing the annotation files.
        output_dir: Directory to merge the images and annotations into.
        file_extension: File extension of the images. Defaults to ".png".
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


def yolo_detect(
    model: YOLO,
    model_name: str,
    image_dir: str = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images",
    output_dir: str = "/workspaces/gorillatracker/data/derived_data/cxl",
    output_task: str = "face_bbox",
    file_extension: str = ".png",
) -> None:
    """Detect gorilla faces in the given directory and save the results in the output directory using the given yolo model."""
    output_dir = os.path.join(output_dir, model_name, output_task)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = model(image_path)

        annotation_path = os.path.join(output_dir, image_file.replace(file_extension, ".txt"))
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        result[0].save_txt(annotation_path, save_conf=True)  # NOTE: simply appends to the .txt file


# if __name__ == "__main__":
# bristol_split_dir = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-0-test-0-mintraincount-3-seed-69-train-70-val-15-test-15"
# model, model_name = modify_dataset_train_yolo(
#     bristol_split_dir,
#     model_type="yolov8x",
#     epochs=2,
#     batch_size=16,
# )

# yolo_detect(
#     model,
#     model_name,
#     output_task="face_bbox",
# )

# model = YOLO("/workspaces/gorillatracker/models/yolov8n_gorillabody_ybyh495y.pt")
# model_name = "yolov8n_gorillabody_ybyh495y"
# yolo_detect(
#     model,
#     model_name,
#     output_task="body_bbox",
# )
