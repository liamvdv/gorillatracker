import os
import cv2
from ultralytics import YOLO
import gorillatracker.utils.visualizer_helpers as visualizer_helpers
import gorillatracker.utils.cutout_helpers as cutout_helpers

from typing import List, Dict, Optional


def _group_images_by_label(images: List[str]) -> Dict[str, List[str]]:
    images_grouped = {}
    for image in images:
        image_group = image[:4]
        if image_group not in images_grouped:
            images_grouped[image_group] = []
        images_grouped[image_group].append(image)
    return images_grouped
        

def _get_labeled_bbox(src_dir: str, label_dir: str, labeled_images: List[str]) -> Dict[str, cutout_helpers.BOUNDING_BOX]:
    labeled_bbox = {}
    for label in labeled_images:
        src_image_path = os.path.join(src_dir, label)
        label_image_path = os.path.join(label_dir, label)
        src_image = cv2.imread(src_image_path)
        label_image = cv2.imread(label_image_path)
        labeled_bbox[label] = cutout_helpers.get_cutout_bbox(src_image, label_image)
    return labeled_bbox

def _create_reference_plot(full_images: str, src_dir: str, bbox: Dict[str, cutout_helpers.BOUNDING_BOX]) -> None:
    images = []
    for image_name in full_images:
        image_path = os.path.join(src_dir, image_name)
        image = cv2.imread(image_path)
        if image_name in bbox:
            visualizer_helpers.draw_bbox(image, bbox[image_name])
        images.append(image)
    visualizer_helpers.create_image_grid(images).savefig("reference_plot.png")
        
def _predict_bbox(image_path: str, model: YOLO) -> List[cutout_helpers.BOUNDING_BOX]:
    result = model(image_path)[0]
    bbox = result.boxes.xyxy.cpu().numpy()
    bbox = bbox.reshape(-1, 2, 2)
    bbox = bbox.astype(int)
    return bbox

def _label_image(image_path, model: YOLO) -> Optional[cutout_helpers.BOUNDING_BOX]:
    image = cv2.imread(image_path)
    bbox = _predict_bbox(image_path, model)
    
    if len(bbox) == 0:
        print(f"WARNING: No bbox found for {image_path}")
        return None
    
    for i, b in enumerate(bbox):
        visualizer_helpers.draw_bbox(image, b, label=str(i))
    cv2.imwrite("to_label.png", image)

    box_id = -1
    while True:
        box_id = int(input("Enter bbox to use: "))
        if box_id in range(len(bbox)):
            break
        elif box_id == -1:
            print(f"No bbox accepted for {image_path}")
            return None
        else:
            print("Invalid bbox id, enter -1 if no bbox is matching")
    return bbox[box_id]
    
    
def _label_group(group: str, full_images_group: List[str], src_dir: str, target_dir: str, model: YOLO) -> None:
    print("#"*80)
    print(f"Labeling group {group}")
    print("#"*80)
    face_images_group = list(filter(lambda x: x.startswith(group), os.listdir(target_dir)))
    to_label = [f for f in full_images_group if f not in face_images_group]
    bbox = _get_labeled_bbox(src_dir, target_dir, face_images_group)
    for image in to_label:    
        _create_reference_plot(full_images_group, src_dir, bbox)
        label = _label_image(os.path.join(src_dir, image), model)
        if label is None:
            continue
        cutout_helpers.cutout_image(cv2.imread(os.path.join(src_dir, image)), label, os.path.join(target_dir, image))
        bbox[image] = label
    
def label_dataset(src_dir: str, target_dir: str, model: YOLO) -> None:
    """
    Tool to label the CXL dataset, task is to crop the gorilla associated with the first 4 letters of the filename.
    
    Args:
    src_dir: Directory containing only the source images as png.
    target_dir: Directory to save the cropped images to. Already labeled images will be skipped for labeling and used as decision aid.
    model_path: YOLO Model to use for detection.
    """
    full_images = os.listdir(src_dir)
    os.makedirs(target_dir, exist_ok=True)
    face_images = os.listdir(target_dir)
    
    assert all([f.endswith(".png") for f in full_images]), "Source directory must only contain png images"
    assert all([f.endswith(".png") for f in face_images]), "Target directory must only contain png images"
    assert all([f in full_images for f in face_images]), "Target directory must be subset of source directory"
    
    full_images_grouped = _group_images_by_label(full_images)
    
    for full_image_group in full_images_grouped:
        _label_group(full_image_group, full_images_grouped[full_image_group], src_dir, target_dir, model)
        
if __name__ == "__main__":
    src_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/full_images"
    target_dir = "/workspaces/gorillatracker/data/ground_truth/cxl/face_images"
    model_path = "/workspaces/gorillatracker/models/yolov8n_gorillaface_pkmbzis.pt"
    model = YOLO(model_path)
    label_dataset(src_dir, target_dir, model)