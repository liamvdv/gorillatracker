import numpy as np
import xml.etree.ElementTree as ET

# taken from https://github.com/opencv/cvat/issues/5828
def rle2Mask(rle: list[int], width: int, height:int)->np.ndarray:
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

def extract_segment_from_mask(mask_element, box_width, box_height)->np.ndarray:
    label = mask_element.get('label')
    assert(label == 'gorilla')
    rle = mask_element.get('rle')
    rle = list(map(int, rle.split(', ')))
    assert(sum(rle) == box_width * box_height)
    mask = rle2Mask(rle, box_width, box_height)
    return mask


def extract_boxes_from_mask(mask_element):
    left = int(mask_element.get('left'))
    top = int(mask_element.get('top'))
    width = int(mask_element.get('width'))
    height = int(mask_element.get('height'))
    
    x_min = left
    y_min = top
    x_max = left + width
    y_max = top + height
    
    return x_min, y_min, x_max, y_max

def cvat_import(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    import_dict = {}

    for image in root.findall('.//image'):
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        img_name = image.get('name')
        filename = img_name.split('.')[0]

        boxes = []
        segments = []
        for mask in image.findall('.//mask'):
            box = extract_boxes_from_mask(mask)
            boxes.append(box)
            segments.append(extract_segment_from_mask(mask, box[2] - box[0], box[3] - box[1]))
            
        if boxes and segments:
            import_dict[filename] = {'boxes': boxes, 'segments': segments, 'width': img_width, 'height': img_height}
            
    return import_dict