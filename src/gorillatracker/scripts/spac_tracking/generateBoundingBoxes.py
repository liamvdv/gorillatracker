from ultralytics import YOLO
import os
import json
    

                  
def createLabels(input_path, models, json_folder = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels", overwrite_json = False):
    fileName = input_path.split("/")[-1]
    fileName = fileName.split(".")[:-1]
    fileName = ".".join(fileName)
    json_path = f"{json_folder}/{fileName}.json"
    if os.path.exists(json_path) and not overwrite_json:
        return
    results = []
    for model in models:
        results.append(model.predict(input_path, stream = True))
    labelFrames = [[] for f in results[0]]
    for resultIndex in range(len(results)):
        result = results[resultIndex]
        frameIndex = 0
        for frame in result:
            boxes = frame.boxes.xywhn.tolist()
            confs = frame.boxes.conf.tolist()
            for box, conf in zip(boxes, confs):
                x, y, w, h = box
                box = {
                    "class": resultIndex,
                    "center_x": x,
                    "center_y": y,
                    "w": w,
                    "h": h,
                    "conf": conf
                }
                labelFrames[frameIndex].append(box)
            frameIndex += 1
    print(f"Saving to {json_path}", end="\r")
    json.dump({"labels": labelFrames}, open(json_path, "w"), indent=4)
    print(f"Saved to {json_path}   ")

    
# yolo label structure: [class, center_x, center_y, w, h, conf], all values are ratios of the image size