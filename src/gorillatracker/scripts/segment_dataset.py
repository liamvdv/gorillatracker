import os

import cv2
import numpy as np
import numpy.typing as npt
from segment_anything import SamPredictor, sam_model_registry

import gorillatracker.type_helper as gtyping
import gorillatracker.utils.cutout_helpers as cutout_helpers

MODEL_PATH = "/workspaces/gorillatracker/models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"


def _predict_mask(
    predictor: SamPredictor, image: gtyping.Image, bbox: gtyping.BoundingBox, image_format: str
) -> npt.NDArray[np.bool_]:
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    box = np.array([x_min, y_min, x_max, y_max])

    predictor.set_image(image, image_format)
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    return mask


def _remove_background(image: gtyping.Image, mask: npt.NDArray[np.bool_]) -> gtyping.Image:
    mask = mask.squeeze()
    assert image.shape[:2] == mask.shape
    background_color = (255, 255, 255)
    image[~mask] = background_color
    return image


def segment_image(image: gtyping.Image, bbox: gtyping.BoundingBox) -> gtyping.Image:
    """
    Args:
        image: (H, W, 3) RGB image
    Returns:
        image: segmented (H, W, 3) RBG image with background white (255, 255, 255)
    """
    return segment_images([image], [bbox])[0]


def segment_images(
    images: list[gtyping.Image], bboxes: list[gtyping.BoundingBox], image_format: str = "RGB"
) -> list[gtyping.Image]:
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
    assert len(images) == len(bboxes)
    for image, bbox in zip(images, bboxes):
        mask = _predict_mask(predictor, image, bbox, image_format)
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
    full_images = [cv2.imread(os.path.join(image_dir, i)) for i in cutout_image_names]
    cutout_images = [cv2.imread(os.path.join(cutout_dir, i)) for i in cutout_image_names]
    assert len(full_images) == len(cutout_images)
    bboxes = [cutout_helpers.get_cutout_bbox(f, c) for f, c in zip(full_images, cutout_images)]

    segmented_images = segment_images(full_images, bboxes, image_format="BGR")
    for name, segment_image, bbox in zip(cutout_image_names, segmented_images, bboxes):
        cutout_helpers.cutout_image(segment_image, bbox, os.path.join(target_dir, name))


if __name__ == "__main__":
    image_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images"
    cutout_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images"
    target_dir = "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/segmented_body_images"
    segment_dir(image_dir, cutout_dir, target_dir)
