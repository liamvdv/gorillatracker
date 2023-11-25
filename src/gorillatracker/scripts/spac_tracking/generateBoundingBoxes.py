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
        print(f"{label}", end="\r")
        labelIndex = int(label.split(".")[0].split("_")[-1]) - 1
        boxes = []
        with open(os.path.join(labels_path, label), "r") as f:
            for line in f.readlines():
                if len(line.split(" ")) < 6:
                    continue
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
    print("/r")
    print("")
def convertLabelsToJsonDebug(input_path):
    vidFrameCount = int(os.popen(f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {input_path}").read().split("\n")[0])
    fileName = input_path.split("/")[-1]
    fileName = fileName.split(".")[:-1]
    fileName = ".".join(fileName) # remove file extension
    tmpPath = f"./mTrack/tmp/{fileName}"
    convertLabelsToJson(f"{tmpPath}/joined", f"./mTrack/{fileName}.json", vidFrameCount)
    

def joinLabels(labels_paths, joined_labels_path, frameCount):
    inputFileName = labels_paths[0].split("/")[-3]
    if not os.path.exists(joined_labels_path):
        os.mkdir(joined_labels_path)
    for f in range(frameCount):
        print(f"{f}/{frameCount}", end="\r")
        for classIndex in range(len(labels_paths)):
            classDirPath = labels_paths[classIndex]
            indices = [int(x.split(".")[0].split("_")[-1]) for x in os.listdir(classDirPath)]
            if f in indices:
                with open(os.path.join(classDirPath, f"{inputFileName}_{f}.txt"), "r") as f1:
                    with open(os.path.join(joined_labels_path, f"{inputFileName}_{f}.txt"), "a") as f2:
                        labels =  f1.read()
                        labels = setLabelClass(labels, classIndex)
                        f2.write(f"{labels}")
    print("")
                        
def setLabelClass(labelFileStr, classIndex):
    labelFileStrLines = labelFileStr.split("\n")
    for i in range(len(labelFileStrLines)-1):
        labelSplit = labelFileStrLines[i].split(" ")
        labelSplit[0] = str(classIndex)
        labelFileStrLines[i] = " ".join(labelSplit)
    return "\n".join(labelFileStrLines)
                  
def createLabels(input_path, bodyModel, faceModel):
    vidFrameCount = int(os.popen(f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {input_path}").read().split("\n")[0])
    fileName = input_path.split("/")[-1]
    fileName = fileName.split(".")[:-1]
    fileName = ".".join(fileName) # remove file extension
    tmpPath = f"./mTrack/tmp/{fileName}"
    bodyModel.predict(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "body")
    faceModel.predict(input_path, save_txt = True, save_conf = True, project = tmpPath, name = "face")
    joinLabels([f"{tmpPath}/body/labels", f"{tmpPath}/face/labels"], f"{tmpPath}/joined", vidFrameCount)
    convertLabelsToJson(f"{tmpPath}/joined", f"mTrack/output/{fileName}.json", vidFrameCount)
    os.system(f"rm -rf {tmpPath}")
    
def joinLabelsDebug(input_path):
    vidFrameCount = int(os.popen(f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {input_path}").read().split("\n")[0])
    fileName = input_path.split("/")[-1]
    fileName = fileName.split(".")[:-1]
    fileName = ".".join(fileName) # remove file extension
    tmpPath = f"./mTrack/tmp/{fileName}"
    joinLabels([f"{tmpPath}/body/labels", f"{tmpPath}/face/labels"], f"{tmpPath}/joined", vidFrameCount)

    
    

if __name__ == "__main__":
    if not os.path.exists("./mTrack"):
        os.mkdir("./mTrack")
    if os.path.exists("./mTrack/tmp"):
        os.system("rm -rf ./mTrack/tmp")
    os.mkdir("./mTrack/tmp")
    
# yolo label structure: [class, center_x, center_y, w, h, conf], all values are ratios of the image size