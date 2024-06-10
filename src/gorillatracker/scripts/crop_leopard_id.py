import json
import os

import cv2
from tqdm import tqdm


def crop_images_from_coco(annotation_path: str, image_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(annotation_path, "r") as f:
        annotation_file = json.load(f)
    annotations = annotation_file.get("annotations", [])
    images = annotation_file.get("images", [])
    image_file_names = {img["id"]: img["file_name"] for img in images}
    for ann in tqdm(annotations, desc="Cropping images", total=len(annotations), unit="images"):
        image_id = ann["image_id"]
        leopard_name = ann["name"]
        if leopard_name.count("-") != 4:
            print(f"Invalid leopard name: {leopard_name}")
            continue

        image_file = image_file_names.get(image_id)
        if image_file is None:
            print(f"Image file not found for image_id: {image_id}")
            continue
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image file: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = ann["bbox"]
        x, y, width, height = bbox
        cropped_image = image[int(y) : int(y + height), int(x) : int(x + width)]

        output_path = os.path.join(output_dir, f"{leopard_name}_{image_id}.png")
        cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    data_dir = "video_data/external-datasets/LeopardID2022/leopard.coco"
    annotation_file = os.path.join(data_dir, "annotations/instances_train2022.json")
    image_dir = os.path.join(data_dir, "images/train2022")
    output_dir = "compressed/test"
    # output_dir = "video_data/external-datasets/LeopardID2022/cropped-images/train"
    crop_images_from_coco(annotation_file, image_dir, output_dir)
