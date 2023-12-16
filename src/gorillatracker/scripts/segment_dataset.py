import os
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from segment_anything import SamPredictor, sam_model_registry

import gorillatracker.utils.cutout_helpers as cutout_helpers

MODEL_PATH = "/workspaces/gorillatracker/models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"

BOUNDING_BOX = Tuple[Tuple[int, int], Tuple[int, int]]


def _predict_mask(predictor: SamPredictor, image: npt.NDArray[np.uint8], bbox: BOUNDING_BOX) -> npt.NDArray[np.uint8]:
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    box = np.array([x_min, y_min, x_max, y_max])

    predictor.set_image(image)
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    return mask


def _remove_background(image: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w = mask.shape[-2:]
    reshaped_mask = mask.reshape(h, w, 1)
    inverted_mask = (1 - reshaped_mask) * 255
    return image * reshaped_mask + inverted_mask


def segment_image(image: npt.NDArray[np.uint8], bbox: BOUNDING_BOX) -> npt.NDArray[np.uint8]:
    """
    Args:
        image: (H, W, 3) RGB image
    Returns:
        image: segmented (H, W, 3) RBG image with background white (255, 255, 255)
    """
    return segment_images([image], [bbox])[0]


def segment_images(images: List[npt.NDArray[np.uint8]], bboxes: List[BOUNDING_BOX]) -> List[npt.NDArray[np.uint8]]:
    """
    Args:
        images: list of (H, W, 3) RGB images
    Returns:
        images: list of segmented (H, W, 3) RBG images with background white (255, 255, 255)
    """
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    segment_images = []
    for image, bbox in zip(images, bboxes):
        mask = _predict_mask(predictor, image, bbox)
        segment_images.append(_remove_background(image, mask))
    return segment_images


def segment_dir(image_dir: str, cutout_dir: str, target_dir: str) -> None:
    """
    Segment all cutout images in cutout_dir and save them to target_dir.
    Image dir should contain the full images that the cutout images were cut from to increase SAM performance.

    Args:
        image_dir: directory containing full images
        cutout_dir: directory containing cutout images
        target_dir: directory to save segmented cutout images to

    """
    cutout_image_names = os.listdir(cutout_dir)
    full_images = [cv2.imread(os.path.join(image_dir, image_name)) for image_name in cutout_image_names]
    cutout_images = [cv2.imread(os.path.join(cutout_dir, image_name)) for image_name in cutout_image_names]
    bboxes = [
        cutout_helpers.get_cutout_bbox(full_image, cutout_image)
        for full_image, cutout_image in zip(full_images, cutout_images)
    ]
    full_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in full_images]

    segmented_images = segment_images(full_images, bboxes)
    for name, segment_image, bbox in zip(cutout_image_names, segmented_images, bboxes):
        cutout_helpers.cutout_image(segment_image, bbox, os.path.join(target_dir, name))


if __name__ == "__main__":
    image_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images"
    cutout_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images"
    target_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/segmented_body_images"
    segment_dir(image_dir, cutout_dir, target_dir)
