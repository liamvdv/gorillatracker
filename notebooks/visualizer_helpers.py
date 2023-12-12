from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_sam_mask(mask, ax, color=np.array([30 / 255, 144 / 255, 255 / 255, 0.6])):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_sam_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2))
    
def show_yolo_box(image_path, bbox_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    with open(bbox_path, "r") as file:
        bboxes = file.readlines()
    
    for bbox in bboxes:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, bbox.split()[:5])
        x_min = int((x_center - bbox_width / 2) * width)
        y_min = int((y_center - bbox_height / 2) * height)
        x_max = int((x_center + bbox_width / 2) * width)
        y_max = int((y_center + bbox_height / 2) * height)
        image = draw_bbox(image, ((x_min, y_min), (x_max, y_max)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
def show_bbox(image: npt.NDArray[np.uint8], bbox: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
    """
    Show a bounding box on an image.
    
    Args:
    image_path: path to image
    bbox: ((x_top_left, y_top_left), (x_bottom_right, y_bottom_right))
    
    """
    image = draw_bbox(image, bbox)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
def draw_bbox(img: npt.NDArray[np.uint8], bbox: Tuple[Tuple[int, int], Tuple[int, int]]) -> npt.NDArray[np.uint8]:
    """
    Show a bounding box on an image.
    
    Args:
    img: cv2 image
    bbox: ((x_top_left, y_top_left), (x_bottom_right, y_bottom_right))
    
    Returns:
    image with bounding box drawn on it
    
    """
    return cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 3)
