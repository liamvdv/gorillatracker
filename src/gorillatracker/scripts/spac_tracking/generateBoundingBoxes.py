from ultralytics import YOLO
import os
import subprocess
import json

#TODO: put weights somewhere less hardcoded
bodyModel = YOLO("./weights/body.pt")
faceModel = YOLO("./weights/face.pt")

def compress_video(input_path, output_path):
    # Run FFmpeg command to compress the video
    command = f"ffmpeg -i {input_path} -vcodec libx264 -crf 28 {output_path}"
    subprocess.call(command, shell=True)
    
def convertLabelsToJson(labels_path, json_path, frameCount):
    labelAry = {"labels": [[] for x in range(frameCount)]}
    for label in os.listdir(labels_path):
        labelIndex = int(label.split(".")[0].split("_")[-1]) - 1
        boxes = []
        with open(os.path.join(labels_path, label), "r") as f:
            for line in f.readlines():
                line = line.split(" ")
                boxes.append({
                    "class": int(line[0]),
                    "center_x": float(line[1]),
                    "center_y": float(line[2]),
                    "w": float(line[3]),
                    "h": float(line[4]),
                    "conf": float(line[5])
                })
        labelAry["labels"][labelIndex] = boxes
    with open(json_path, "w") as f:
        json.dump(labelAry, f, indent=4)

def joinLabels(labels_paths, joined_labels_path, frameCount):
    inputFileName = labels_paths[0].split("/")[-1].split("_")[:-1].join("_")
    for f in frameCount:
        for classIndex in range(labels_paths):
            classDirPath = labels_paths[classIndex]
            indices = [int(x.split(".")[0].split("_")[-1]) for x in os.listdir(classDirPath)]
            if f in indices:
                with open(os.path.join(classDirPath, f"{inputFileName}_{f}.txt"), "r") as f:
                    with open(os.path.join(joined_labels_path, f"{inputFileName}_{f}.txt"), "a") as f2:
                        f2.write(f"\n{f.read()}")
                  
def createLabels(input_path, bodyModel, faceModel):
    vidFrameCount = int(os.popen(f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {input_path}").read().split("\n")[0])
    fileName = input_path.split("/")[-1].split(".")[:-1].join(".") # remove file extension
    tmpPath = f"./mTrack/tmp/{fileName}"
    bodyModel.predict(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "body")
    faceModel.predict(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "face")
    joinLabels([f"{tmpPath}/body", f"{tmpPath}/face"], f"{tmpPath}/joined", vidFrameCount)
    convertLabelsToJson(f"{tmpPath}/joined", f"{tmpPat}/{fileName}.json", vidFrameCount)
    
    

    
    

if __name__ == "__main__":
    if not os.path.exists("./mTrack"):
        os.mkdir("./mTrack")
        os.mkdir("./mTrack/output")
    if os.path.exists("./mTrack/tmp"):
        os.system("rm -rf ./mTrack/tmp")
    os.mkdir("./mTrack/tmp")
    
# yolo label structure: [class, center_x, center_y, w, h, conf], all values are ratios of the image size