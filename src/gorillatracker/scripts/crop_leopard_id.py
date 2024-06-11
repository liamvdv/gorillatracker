import json
import os

import cv2
from tqdm import tqdm


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

def extract_image_file_names(images: list) -> dict:
    return {img["id"]: img["file_name"] for img in images}

def is_valid_leopard_name(name: str) -> bool:
    return name.count("-") == 4

def load_image(image_path: str) -> cv2.Mat:
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image file: {image_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop_image(image: cv2.Mat, bbox: list) -> cv2.Mat:
    x, y, width, height = bbox
    return image[int(y) : int(y + height), int(x) : int(x + width)]

def filter_duplicate_annotations(annotations: list) -> list:
    image_ids = [ann["image_id"] for ann in annotations]
    duplicate_image_ids = [image_id for image_id in image_ids if image_ids.count(image_id) > 1]

    return [ann for ann in annotations if ann["image_id"] not in duplicate_image_ids]

def crop_images_from_coco(annotation_path: str, image_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    annotation_file = load_json(annotation_path)
    annotations = annotation_file.get("annotations", [])
    # There are images with multiple annotations and different names which need to be filtered out
    annotations = filter_duplicate_annotations(annotations)
    images = annotation_file.get("images", [])
    image_file_names = extract_image_file_names(images)
    for ann in tqdm(annotations, desc="Cropping images", total=len(annotations), unit="image"):
        image_id = ann["image_id"]
        leopard_name = ann["name"]
        if not is_valid_leopard_name(leopard_name):
            print(f"Invalid leopard name: {leopard_name}")
            continue

        image_file = image_file_names.get(image_id)
        if image_file is None:
            print(f"Image file not found for image_id: {image_id}")
            continue

        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        if image is None:
            continue

        cropped_image = crop_image(image, ann["bbox"])
        output_path = os.path.join(output_dir, f"{leopard_name}_{image_id}.png")
        cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    data_dir = "/workspaces/gorillatracker/datasets/vast/external-datasets/LeopardID2022/leopard.coco"
    annotation_file = os.path.join(data_dir, "annotations/instances_train2022.json")
    image_dir = os.path.join(data_dir, "images/train2022")
    output_dir = "/workspaces/gorillatracker/datasets/vast/external-datasets/LeopardID2022/cropped_images/train"
    crop_images_from_coco(annotation_file, image_dir, output_dir)
