import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
from segment_anything import SamPredictor, sam_model_registry

MODEL_PATH = "/workspaces/gorillatracker/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"


def _predict_mask(predictor: SamPredictor, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    box = np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.uint32)
    predictor.set_image(image)
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
    )
    return mask


def _remove_background(image: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w = mask.shape[-2:]
    reshaped_mask = mask.reshape(h, w, 1)
    inverted_mask = (1 - reshaped_mask) * 255
    return image * reshaped_mask + inverted_mask


def predict_full_image_mask(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    return [_predict_mask(predictor, image) for image in images]


def segment_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Args:
        image: (H, W, 3) RGB image
    Returns:
        image: segmented (H, W, 3) RBG image with background white (255, 255, 255)
    """
    return segment_images([image])[0]


def segment_images(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
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
    for image in images:
        mask = _predict_mask(predictor, image)
        segment_images.append(_remove_background(image, mask))
    return segment_images


def segment_dir(image_dir: str, target_dir: str) -> None:
    """
    Args:
        image_dir: path to directory containing images
        target_dir: path to directory to save segmented images
    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    for image_path in Path(image_dir).glob("*.png"):
        image = cv2.imread(str(image_path))
        segmented_image = segment_image(image)
        cv2.imwrite(str(Path(target_dir) / image_path.name), segmented_image)


if __name__ == "__main__":
    image_dir = "/workspaces/gorillatracker/data/derived_data/rohan-cxl/filtered_body_images"
    target_dir = "/workspaces/gorillatracker/data/derived_data/rohan-cxl/filtered_segmented_body_images"
    segment_dir(image_dir, target_dir)
