import os
from typing import List, Tuple

import cv2

import gorillatracker.utils.cutout_helpers as cutout_helpers
import gorillatracker.utils.yolo_helpers as yolo_helpers

BOUNDING_BOX = Tuple[Tuple[int, int], Tuple[int, int]]


def calculate_area(box: BOUNDING_BOX) -> float:
    x_min, y_min = box[0]
    x_max, y_max = box[1]
    return (x_max - x_min) * (y_max - y_min)


def calculate_intersection_area(box1: BOUNDING_BOX, box2: BOUNDING_BOX) -> float:
    x_left = max(box1[0][0], box2[0][0])
    x_right = min(box1[1][0], box2[1][0])
    y_top = max(box1[0][1], box2[0][1])
    y_bottom = min(box1[1][1], box2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def find_max_intersection(target_bbox: BOUNDING_BOX, bboxes: List[BOUNDING_BOX]) -> BOUNDING_BOX:
    max_area = 0.0
    max_bbox = None

    for bbox in bboxes:
        area = calculate_intersection_area(target_bbox, bbox)
        if area > max_area:
            max_area = area
            max_bbox = bbox

    if max_area / calculate_area(target_bbox) < 0.5:  # NOTE(memben) target_box should be a subset of max_bbox
        print(f"Warning: max intersection area is only {max_area / calculate_area(target_bbox)} of target area")

    assert max_bbox is not None, "No intersection found"
    return max_bbox


def expand_bounding_box(bbox_to_expand: BOUNDING_BOX, bbox_to_include: BOUNDING_BOX) -> BOUNDING_BOX:
    x_min = min(bbox_to_include[0][0], bbox_to_expand[0][0])
    y_min = min(bbox_to_include[0][1], bbox_to_expand[0][1])
    x_max = max(bbox_to_include[1][0], bbox_to_expand[1][0])
    y_max = max(bbox_to_include[1][1], bbox_to_expand[1][1])
    return ((x_min, y_min), (x_max, y_max))


def cutout_with_integrity(
    full_image_path: str, cutout_path: str, bbox_file_path: str, target_path: str, force_include: bool
) -> None:
    full_image = cv2.imread(full_image_path)
    cutout = cv2.imread(cutout_path)
    target_bbox = cutout_helpers.get_cutout_bbox(full_image, cutout)
    bboxes = yolo_helpers.convert_annotation_file(bbox_file_path, full_image.shape[1], full_image.shape[0])
    assert bboxes, f"No bounding boxes found in {bbox_file_path}"
    max_bbox = find_max_intersection(target_bbox, bboxes)

    if force_include:
        max_bbox = expand_bounding_box(target_bbox, max_bbox)

    cutout_helpers.cutout_image(full_image_path, max_bbox, target_path)


def cutout_dataset_with_integrity(
    full_image_dir: str, cutout_dir: str, bbox_dir: str, target_dir: str, force_include: bool = False
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for cutout_file in os.listdir(cutout_dir):
        cutout_path = os.path.join(cutout_dir, cutout_file)
        full_image_path = os.path.join(full_image_dir, cutout_file)
        bbox_path = os.path.join(bbox_dir, cutout_file.replace(".png", ".txt"))
        target_path = os.path.join(target_dir, cutout_file)
        assert os.path.exists(full_image_path), f"Full image file {full_image_path} does not exist"
        assert os.path.exists(bbox_path), f"Annotation file {bbox_path} does not exist"
        cutout_with_integrity(full_image_path, cutout_path, bbox_path, target_path, force_include)


if __name__ == "__main__":
    cutout_dataset_with_integrity(
        "/workspaces/gorillatracker/data/ground_truth/cxl/full_images",
        "/workspaces/gorillatracker/data/ground_truth/cxl/face_images",
        "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_bbox",
        "/workspaces/gorillatracker/data/derived_data/cxl/yolov8n_gorillabody_ybyh495y/body_images",
        force_include=True,
    )
