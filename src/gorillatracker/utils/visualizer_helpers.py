import math
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import gorillatracker.type_helper as gtyping
from gorillatracker.utils.yolo_helpers import convert_from_yolo_format


# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_sam_mask(
    mask: npt.NDArray[np.uint8],
    ax: Axes,
    color: npt.NDArray[np.float32] = np.array([30 / 255, 144 / 255, 255 / 255, 0.6]),
) -> None:
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_sam_box(box: gtyping.BoundingBox, ax: Axes) -> None:
    x_min, y_min = box[0]
    x_max, y_max = box[1]
    w = x_max - x_min
    h = y_max - y_min
    ax.add_patch(Rectangle((x_min, y_min), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2))


def show_yolo_box(image_path: str, bbox_path: str) -> None:
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    with open(bbox_path, "r") as file:
        bboxes = file.readlines()

    for bbox_str in bboxes:
        _, *yolo_bbox = list(map(float, bbox_str.split()))[:5]
        bbox = convert_from_yolo_format(yolo_bbox, width, height)
        image = draw_bbox(image, bbox)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def show_image_on_remote(image: gtyping.Image) -> None:
    """
    Show a cv2 image.

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("my_plot.png")


def show_bbox(image: gtyping.Image, bbox: gtyping.BoundingBox) -> None:
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


def draw_bbox(img: gtyping.Image, bbox: gtyping.BoundingBox, label: Optional[str] = None) -> gtyping.Image:
    """
    Show a bounding box on an image.

    Args:
    img: cv2 image
    bbox: ((x_top_left, y_top_left), (x_bottom_right, y_bottom_right))

    Returns:
    image with bounding box drawn on it

    """
    red = (0, 0, 255)  # BGR
    if label:
        cv2.putText(img, label, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    return cv2.rectangle(img, bbox[0], bbox[1], red, 3)


def create_image_grid(images: list[gtyping.Image], width: int = 3) -> Figure:
    """
    Creates a grid of images.
    """
    if len(images) == 0:
        print("WARNING: No images to plot!")
        return plt.figure()

    height = math.ceil(len(images) / width)
    fig, axs = plt.subplots(height, width, figsize=(20, 40))

    # Ensure axs is a 2D array
    axs = np.array(axs).reshape(height, width)

    for i, image in enumerate(images):
        ax = axs[i // width, i % width]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

    for ax in axs.ravel():
        ax.set_axis_off()

    plt.tight_layout()
    return fig
