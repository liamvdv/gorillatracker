import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# NOTE(liamvdv): Will exclude from mypy.
# TODO(memben): MyPy annotate this file, then add back to pyproject.toml [mypy]


@dataclass
class SegmentedImageData:
    path: str
    segments: Dict[str, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = field(default_factory=dict)

    def add_segment(self, class_label: str, mask: np.ndarray, box: Tuple[int, int, int, int]):
        """
        class_label: label of the segment
        mask: binary mask of the segment
        box: bounding box of the segment in x_min, y_min, x_max, y_max format
        """
        if class_label not in self.segments:
            self.segments[class_label] = []
        self.segments[class_label].append((mask, box))

    @property
    def filename(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


# taken from https://github.com/opencv/cvat/issues/5828 and modified
def _rle2Mask(rle: list[int], width: int, height: int) -> np.ndarray:
    decoded = np.zeros(width * height, dtype=np.uint8)
    pos = 0
    for i, val in enumerate(rle):
        decoded[pos : pos + val] = i % 2
        pos += val
    return decoded.reshape((height, width))


def _extract_segment_from_mask_element(mask_element, box_width, box_height) -> np.ndarray:
    label = mask_element.get("label")
    assert label == "gorilla"
    rle = mask_element.get("rle")
    rle = list(map(int, rle.split(", ")))
    assert sum(rle) == box_width * box_height
    mask = _rle2Mask(rle, box_width, box_height)
    return mask


def _expand_segment_to_img_mask(segment, img_width, img_height, box_x_min, box_y_min):
    mask = np.zeros((img_height, img_width), dtype=bool)
    y_max, x_max = box_y_min + segment.shape[0], box_x_min + segment.shape[1]
    mask[box_y_min:y_max, box_x_min:x_max] = segment.astype(bool)
    return mask


def _extract_boxes_from_mask(mask_element):
    left = int(mask_element.get("left"))
    top = int(mask_element.get("top"))
    width = int(mask_element.get("width"))
    height = int(mask_element.get("height"))

    x_min = left
    y_min = top
    x_max = left + width
    y_max = top + height

    return x_min, y_min, x_max, y_max


def cvat_import(xml_file: str, img_path: str, skip_no_mask=True) -> List[SegmentedImageData]:
    """
    xml_file: path to xml file
    img_path: path to images
    skip_no_mask: if True, skip images with no mask
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    segmented_images = []

    for image in root.findall(".//image"):
        img_width = int(image.get("width"))
        img_height = int(image.get("height"))
        img_name = image.get("name")
        path = img_path + "/" + img_name

        segmented_image = SegmentedImageData(path=path)

        for mask in image.findall(".//mask"):
            label = mask.get("label")
            box = _extract_boxes_from_mask(mask)
            box_mask = _extract_segment_from_mask_element(mask, box[2] - box[0], box[3] - box[1])
            img_mask = _expand_segment_to_img_mask(box_mask, img_width, img_height, box[0], box[1])
            segmented_image.add_segment(label, img_mask, box)

        if segmented_image.segments or not skip_no_mask:
            segmented_images.append(segmented_image)

    return segmented_images
