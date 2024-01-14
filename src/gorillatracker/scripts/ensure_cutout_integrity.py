# mypy: warn_unused_ignores=False

import os

import cv2
import numpy as np

import gorillatracker.type_helper as gtyping
import gorillatracker.utils.cutout_helpers as cutout_helpers
import gorillatracker.utils.yolo_helpers as yolo_helpers


def _is_cutout_in_image(image: gtyping.Image, cutout: gtyping.Image) -> bool:
    res = cv2.matchTemplate(image, cutout, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)  # type: ignore
    return len(loc[0]) > 0


def assert_matching_cutouts(image_dir: str, cutout_dir: str) -> None:
    cutout_files = os.listdir(cutout_dir)
    image_files = os.listdir(image_dir)

    assert all([f.endswith(".png") for f in cutout_files])
    assert all([f.endswith(".png") for f in image_files])

    assert len(set(image_files) - set(cutout_files)) == 0
    assert len(set(cutout_files) - set(image_files)) == 0

    for cutout_file in cutout_files:
        image_path = os.path.join(image_dir, cutout_file)
        cutout_path = os.path.join(cutout_dir, cutout_file)

        image = cv2.imread(image_path)
        cutout = cv2.imread(cutout_path)

        assert cutout.shape <= image.shape, f"{cutout_file} has larger shape than corresponding image"
        assert _is_cutout_in_image(image, cutout), f"{cutout_file} not in corresponding image"


def calculate_area(box: gtyping.BoundingBox) -> float:
    x_min, y_min = box[0]
    x_max, y_max = box[1]
    return (x_max - x_min) * (y_max - y_min)


def calculate_intersection_area(box1: gtyping.BoundingBox, box2: gtyping.BoundingBox) -> float:
    x_left = max(box1[0][0], box2[0][0])
    x_right = min(box1[1][0], box2[1][0])
    y_top = max(box1[0][1], box2[0][1])
    y_bottom = min(box1[1][1], box2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def find_max_intersection(target_bbox: gtyping.BoundingBox, bboxes: list[gtyping.BoundingBox]) -> gtyping.BoundingBox:
    max_bbox = max(bboxes, key=lambda bbox: calculate_intersection_area(target_bbox, bbox))
    max_area = calculate_intersection_area(target_bbox, max_bbox)
    assert max_area > 0, "No intersection found"
    if max_area / calculate_area(target_bbox) < 0.5:  # NOTE(memben) target_box should be a subset of max_bbox
        print(f"Warning: max intersection area is only {max_area / calculate_area(target_bbox)} of target area")
    return max_bbox


def expand_bounding_box(
    bbox_to_expand: gtyping.BoundingBox, bbox_to_include: gtyping.BoundingBox
) -> gtyping.BoundingBox:
    x_min = min(bbox_to_include[0][0], bbox_to_expand[0][0])
    y_min = min(bbox_to_include[0][1], bbox_to_expand[0][1])
    x_max = max(bbox_to_include[1][0], bbox_to_expand[1][0])
    y_max = max(bbox_to_include[1][1], bbox_to_expand[1][1])
    return ((x_min, y_min), (x_max, y_max))


def cutout(full_image_path: str, cutout_path: str, bbox_file_path: str, target_path: str, expand_bbox: bool) -> None:
    full_image = cv2.imread(full_image_path)
    cutout = cv2.imread(cutout_path)
    target_bbox = cutout_helpers.get_cutout_bbox(full_image, cutout)
    bboxes = yolo_helpers.convert_annotation_file(bbox_file_path, full_image.shape[1], full_image.shape[0])
    assert bboxes, f"No bounding boxes found in {bbox_file_path}"
    max_bbox = find_max_intersection(target_bbox, bboxes)

    if expand_bbox:
        max_bbox = expand_bounding_box(target_bbox, max_bbox)

    cutout_helpers.cutout_image(full_image, max_bbox, target_path)


def cutout_dataset(
    full_image_dir: str, cutout_dir: str, bbox_dir: str, target_dir: str, expand_bbox: bool = False
) -> None:
    """
    Given a full_image and a cutout of the full image (e.g. face) and bounding boxes of the full image (e.g. body).
    Find the bouding box with the highest intersection area with the cutout and cutout the full image with this bounding box.

    Args:
        full_image_dir: Directory containing the full images.
        cutout_dir: Directory containing the cutouts.
        bbox_dir: Directory containing the bounding boxes.
        target_dir: Directory to save the cutout images to.
        expand_bbox: Whether to expand the best bounding box to include the cutout.
    """
    os.makedirs(target_dir, exist_ok=True)
    for cutout_file in os.listdir(cutout_dir):
        cutout_path = os.path.join(cutout_dir, cutout_file)
        full_image_path = os.path.join(full_image_dir, cutout_file)
        bbox_path = os.path.join(bbox_dir, cutout_file.replace(".png", ".txt"))
        target_path = os.path.join(target_dir, cutout_file)
        assert os.path.exists(full_image_path), f"Full image file {full_image_path} does not exist"
        assert os.path.exists(bbox_path), f"Annotation file {bbox_path} does not exist"
        cutout(full_image_path, cutout_path, bbox_path, target_path, expand_bbox)


if __name__ == "__main__":
    # cutout_dataset(
    #     "/workspaces/gorillatracker/data/ground_truth/cxl/full_images",
    #     "/workspaces/gorillatracker/data/ground_truth/cxl/face_images",
    #     "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_bbox",
    #     "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images",
    #     expand_bbox=True,
    # )
    assert_matching_cutouts(
        "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images",
        "/workspaces/gorillatracker/data/ground_truth/cxl/face_images",
    )
