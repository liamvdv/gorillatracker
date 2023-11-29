import multiprocessing
from generateBoundingBoxes import createLabels
import os
from ultralytics import YOLO

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton. # TODO UNTESTED
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if multiprocessing.current_process().pid not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[multiprocessing.current_process().pid] = instance
        return cls._instances[multiprocessing.current_process().pid]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        bodyModel = YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/body.pt")
        faceModel = YOLO("/workspaces/gorillatracker/src/gorillatracker/scripts/spac_tracking/weights/body.pt")
        self.value = (bodyModel, faceModel)

    def get_yolo(self): 
        return self.value

def worker_function(i):
    singleton = Singleton()
    videoPath = i
    bodyModel, faceModel = singleton.get_yolo()
    createLabels(videoPath, bodyModel, faceModel)
    
    

if __name__ == "__main__":
    if not os.path.exists("./mTrack"):
        os.mkdir("./mTrack")
        os.mkdir("./mTrack/output")
    if os.path.exists("./mTrack/tmp"):
        os.system("rm -rf ./mTrack/tmp")
    else:
        os.mkdir("./mTrack/tmp")
    videoDir = "/workspaces/gorillatracker/spac_gorillas_converted"
    print(os.listdir(videoDir))
    videoPaths = [os.path.join(videoDir, x) for x in os.listdir(videoDir)]
    debugVideoPaths = videoPaths[:100]
    debugVideoPaths2 = videoPaths[100:200]
    pool = multiprocessing.Pool(6)
    pool.map(worker_function, debugVideoPaths2)
    pool.close()
    pool.join()