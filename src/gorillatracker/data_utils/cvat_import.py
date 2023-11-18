from typing import List

import numpy as np
import xml.etree.ElementTree as ET

from data_utils.segmented_image_data import SegmentedImageData

# taken from https://github.com/opencv/cvat/issues/5828
def _rle2Mask(rle: list[int], width: int, height:int)->np.ndarray:
    decoded = [0] * (width * height) # create bitmap container
    decoded_idx = 0
    value = 0

    for v in rle:
        decoded[decoded_idx:decoded_idx+v] = [value] * v
        decoded_idx += v
        value = 1 - value # alternate 1/0 for decoding

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((height, width)) # reshape to image size
    return decoded

def _extract_segment_from_mask_element(mask_element, box_width, box_height)->np.ndarray:
    label = mask_element.get('label')
    assert(label == 'gorilla')
    rle = mask_element.get('rle')
    rle = list(map(int, rle.split(', ')))
    assert(sum(rle) == box_width * box_height)
    mask = _rle2Mask(rle, box_width, box_height)
    return mask

def _expand_segment_to_img_mask(segment, img_width, img_height, box_x_min, box_y_min):
    mask = np.zeros((img_height, img_width), dtype=bool)
    y_max, x_max = box_y_min + segment.shape[0], box_x_min + segment.shape[1]
    mask[box_y_min:y_max, box_x_min:x_max] = segment.astype(bool)
    return mask

def _extract_boxes_from_mask(mask_element):
    left = int(mask_element.get('left'))
    top = int(mask_element.get('top'))
    width = int(mask_element.get('width'))
    height = int(mask_element.get('height'))
    
    x_min = left
    y_min = top
    x_max = left + width
    y_max = top + height
    
    return x_min, y_min, x_max, y_max

def cvat_import(xml_file, skip_no_mask=True)-> List[SegmentedImageData]:
    """
    xml_file: path to xml file
    skip_no_mask: if True, skip images with no mask
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    import_dict = {}
    
    segmented_images = []

    for image in root.findall('.//image'):
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        img_name = image.get('name')
        filename = img_name.split('.')[0]

        segmented_image = SegmentedImageData(filename=filename, width=img_width, height=img_height)
        
        for mask in image.findall('.//mask'):
            label = mask.get('label')
            box = _extract_boxes_from_mask(mask)
            box_mask = _extract_segment_from_mask_element(mask, box[2] - box[0], box[3] - box[1])
            img_mask = _expand_segment_to_img_mask(box_mask, img_width, img_height, box[0], box[1])
            segmented_image.add_segment(label, img_mask, box)
            
        if segmented_image.segments or not skip_no_mask:
            segmented_images.append(segmented_image)
            
    return import_dict