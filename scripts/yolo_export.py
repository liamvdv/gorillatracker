# NOTE: https://github.com/ultralytics/ultralytics/issues/5800
# also did not work for the bbox export

import gorillatracker.data_utils.cvat_import as cvat_import 

def _convert_to_yolo_format(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

def _write_box_to_file(filename, boxes):
    with open(filename, "w") as file:
        for box in boxes:
            file.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")

def export_cvat_to_yolo(xml_file, target_dir):
    export = cvat_import.cvat_import(xml_file)
    for filename, data in export.items():
        boxes = data["boxes"]
        img_width = data["width"]
        img_height = data["height"]
        yolo_boxes = [_convert_to_yolo_format(box, img_width, img_height) for box in boxes]
        _write_box_to_file(f"{target_dir}/{filename}.txt", yolo_boxes)    