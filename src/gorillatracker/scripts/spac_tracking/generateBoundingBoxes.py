from ultralytics import YOLO
import os
import subprocess
import json



def compress_video(input_path, output_path):
    # Run FFmpeg command to compress the video
    command = f"ffmpeg -i {input_path} -vcodec libx264 -crf 28 {output_path}"
    subprocess.call(command, shell=True)
    

                  
def createLabels(input_path, bodyModel, faceModel, json_folder = "/workspaces/gorillatracker/data/derived_data/spac_gorillas_converted_labels"):
    fileName = input_path.split("/")[-1]
    fileName = fileName.split(".")[:-1]
    fileName = ".".join(fileName) # remove file extension
    json_path = f"{json_folder}/{fileName}.json"
    if os.path.exists(json_path):
        return
    tmpPath = f"./mTrack/tmp/{fileName}"
    bodyResult = bodyModel.track(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "body", stream = True)
    faceResult = faceModel.track(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "face", stream = True)
    labelFrames = []
    for f, b in zip(faceResult, bodyResult):
        labelFrame = []
        fBoxes = f.boxes.xywhn.tolist()
        bBoxes = b.boxes.xywhn.tolist()
        fConf = f.boxes.conf.tolist()
        bConf = b.boxes.conf.tolist()
        for fb, fc in zip(fBoxes, fConf):
            fx, fy, fw, fh = fb
            box = {
                "class": 0,
                "center_x": fx,
                "center_y": fy,
                "w": fw,
                "h": fh,
                "conf": fc
            }
            labelFrame.append(box)
        for bb, bc in zip(bBoxes, bConf):
            bx, by, bw, bh = bb
            box = {
                "class": 1,
                "center_x": bx,
                "center_y": by,
                "w": bw,
                "h": bh,
                "conf": bc
            }
            labelFrame.append(box)
        labelFrames.append(labelFrame)
    json.dump({"labels": labelFrames}, open(json_path, "w"), indent=4)
    os.system(f"rm -rf {tmpPath}")
    
    

if __name__ == "__main__":
    if not os.path.exists("./mTrack"):
        os.mkdir("./mTrack")
    if os.path.exists("./mTrack/tmp"):
        os.system("rm -rf ./mTrack/tmp")
    os.mkdir("./mTrack/tmp")
    
# yolo label structure: [class, center_x, center_y, w, h, conf], all values are ratios of the image size