"""Scripts to crop the images in the bristol dataset using the bounding boxes provided by the dataset."""

import logging
import os
from typing import List, Literal, Tuple

from PIL import Image

from gorillatracker.scripts.ensure_integrity_openset import bristol_index_to_name

logger = logging.getLogger(__name__)


def crop_and_save_image(image_path: str, x: float, y: float, w: float, h: float, output_path: str) -> None:
    """Crop the image at the given path using the given bounding box coordinates and save it to the given output path.

    Args:
        image_path: Path to the image to crop.
        x: Relative x coordinate of the center of the bounding box.
        y: Relative y coordinate of the center of the bounding box.
        w: Relative width of the bounding box.
        h: Relative height of the bounding box.
        output_path: Path to save the cropped image to.
    """
    img = Image.open(image_path)

    # calculate the bounding box coordinates
    img_width, img_height = img.size
    left = int((x - w / 2) * img_width)
    right = int((x + w / 2) * img_width)
    top = int((y - h / 2) * img_height)
    bottom = int((y + h / 2) * img_height)

    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path)


def read_bbox_data(bbox_path: str) -> List[List[float]]:
    """Read the bounding box data from the given file.

    Args:
        bbox_path: Path to the bounding box file.

    Returns:
        List of bounding box data lines."""
    if not os.path.exists(bbox_path):
        logger.warning("no bounding box found for image with path %s", bbox_path)
        return []

    bbox_data_lines = []
    with open(bbox_path, "r") as bbox_file:
        bbox_data_lines = bbox_file.read().strip().split("\n")

    bbox_data_lines = [line for line in bbox_data_lines if line != ""]
    bbox_data_lines_split = [list(map(float, bbox_data_line.strip().split(" "))) for bbox_data_line in bbox_data_lines]

    return bbox_data_lines_split


def crop_ground_truth(image_path: str, bbox_path: str, output_dir: str) -> None:
    """Crops a single image from the bristol dataset."""
    bbox_data_lines = read_bbox_data(bbox_path)

    for index, x, y, w, h in bbox_data_lines:
        name = bristol_index_to_name[int(index)]
        file_name = name + "_" + os.path.basename(image_path)
        output_path = os.path.join(output_dir, file_name)
        crop_and_save_image(image_path, x, y, w, h, output_path)


def crop_max_confidence(
    image_path: str, bbox_path: str, output_dir: str
) -> Literal["bbox", "no_bbox", "low_confidence"]:
    """Crops a single image from the cxl dataset.
    NOTE: There is only one bounding box per image. Therefore, only the bounding box with the highest confidence score is used.
    NOTE: The confidence score should additionally be at least 0.5.
    """
    bbox_data_lines = read_bbox_data(bbox_path)
    bbox_max_confidence = max(
        bbox_data_lines, key=lambda x: x[-1], default=[-1, -1, -1, -1, -1, -1]
    )  # get the bbox with the highest confidence score

    if bbox_max_confidence[5] >= 0.5:
        _, x, y, w, h, _ = bbox_max_confidence
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        crop_and_save_image(image_path, x, y, w, h, output_path)
        return "bbox"
    elif bbox_max_confidence[0] > 0.0:
        logger.warning("bounding box with confidence score %f is too low for image %s", bbox_max_confidence, image_path)
        return "low_confidence"
    else:
        logger.warning("no bounding box found for image %s predicted", image_path)
        return "no_bbox"


def crop_images(
    image_dir: str, bbox_dir: str, output_dir: str, file_extension: str = ".jpg", is_bristol: bool = True
) -> Tuple[List[str], List[str], List[str]]:  # TODO(rob2u): split into two functions for bristol and cxl
    """Crop all images in the given directory using the bounding boxes in the given directory and save them to the given output directory.

    Args:
        image_dir: Directory containing the images.
        bbox_dir: Directory containing the bounding box files.
        output_dir: Directory to save the cropped images to.
        file_extension: File extension of the images. Defaults to ".jpg".
        is_bristol: Whether the images are from the bristol dataset. Defaults to True.

    Returns:
        Tuple containing the following lists:
            - List of images without a bounding box annotation
            - List of images with no bounding box prediction
            - List of images with a low bounding box confidence score
    """

    os.makedirs(output_dir, exist_ok=True)

    imgs_without_bbox_annotation = []
    imgs_with_no_bbox_pred = []
    imgs_with_low_bbox_confidence = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(file_extension, ".txt"))

        if not os.path.exists(bbox_path):
            logger.warn("no bounding box found for image %s", image_file)
            imgs_without_bbox_annotation.append(image_file)
            continue

        if is_bristol:
            crop_ground_truth(image_path, bbox_path, output_dir)
        else:
            res = crop_max_confidence(image_path, bbox_path, output_dir)
            if res == "no_bbox":
                imgs_with_no_bbox_pred.append(image_file)
            elif res == "low_confidence":
                imgs_with_low_bbox_confidence.append(image_file)

    return imgs_without_bbox_annotation, imgs_with_no_bbox_pred, imgs_with_low_bbox_confidence


if __name__ == "__main__":
    cxl_model_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8x-e30-b163"
    cxl_imgs_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images"

    cxl_imgs_crop_dir = os.path.join(cxl_model_dir, "face_crop")
    cxl_annotation_dir = os.path.join(cxl_model_dir, "face_bbox")

    assert os.path.exists(
        cxl_annotation_dir
    ), f"Bounding box directory '{cxl_annotation_dir}' does not exist, run detect_gorillafaces_cxl from train_yolo.py first"

    os.makedirs(cxl_imgs_crop_dir, exist_ok=True)

    # crop cxl images according to predicted bounding boxes
    imgs_without_bbox, imgs_with_no_bbox_prediction, imgs_with_low_confidence = crop_images(
        cxl_imgs_dir, cxl_annotation_dir, cxl_imgs_crop_dir, is_bristol=False, file_extension=".png"
    )
